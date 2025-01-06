package redis

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/url"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9/internal/pool"
)

// Limiter is the interface of a rate limiter or a circuit breaker.
type Limiter interface {
	// Allow returns nil if operation is allowed or an error otherwise.
	// If operation is allowed client must ReportResult of the operation
	// whether it is a success or a failure.
	Allow() error
	// ReportResult reports the result of the previously allowed operation.
	// nil indicates a success, non-nil error usually indicates a failure.
	ReportResult(result error)
}

// Options keeps the settings to set up redis connection.
type Options struct {
	// The network type, either tcp or unix.
	// Default is tcp.
	Network string
	// host:port address.
	Addr string

	// ClientName will execute the `CLIENT SETNAME ClientName` command for each conn.
	ClientName string

	// Dialer creates new network connection and has priority over
	// Network and Addr options.
	Dialer func(ctx context.Context, network, addr string) (net.Conn, error)

	// Hook that is called when new connection is established.
	OnConnect func(ctx context.Context, cn *Conn) error

	// Protocol 2 or 3. Use the version to negotiate RESP version with redis-server.
	// Default is 3.
	Protocol int
	// Use the specified Username to authenticate the current connection
	// with one of the connections defined in the ACL list when connecting
	// to a Redis 6.0 instance, or greater, that is using the Redis ACL system.
	Username string
	// Optional password. Must match the password specified in the
	// requirepass server configuration option (if connecting to a Redis 5.0 instance, or lower),
	// or the User Password when connecting to a Redis 6.0 instance, or greater,
	// that is using the Redis ACL system.
	Password string
	// CredentialsProvider allows the username and password to be updated
	// before reconnecting. It should return the current username and password.
	CredentialsProvider func() (username string, password string)

	// CredentialsProviderContext is an enhanced parameter of CredentialsProvider,
	// done to maintain API compatibility. In the future,
	// there might be a merge between CredentialsProviderContext and CredentialsProvider.
	// There will be a conflict between them; if CredentialsProviderContext exists, we will ignore CredentialsProvider.
	CredentialsProviderContext func(ctx context.Context) (username string, password string, err error)

	// Database to be selected after connecting to the server.
	DB int

	// Maximum number of retries before giving up.
	// Default is 3 retries; -1 (not 0) disables retries.
	MaxRetries int
	// Minimum backoff between each retry.
	// Default is 8 milliseconds; -1 disables backoff.
	MinRetryBackoff time.Duration
	// Maximum backoff between each retry.
	// Default is 512 milliseconds; -1 disables backoff.
	MaxRetryBackoff time.Duration

	// Dial timeout for establishing new connections.
	// Default is 5 seconds.
	DialTimeout time.Duration
	// Timeout for socket reads. If reached, commands will fail
	// with a timeout instead of blocking. Supported values:
	//   - `0` - default timeout (3 seconds).
	//   - `-1` - no timeout (block indefinitely).
	//   - `-2` - disables SetReadDeadline calls completely.
	ReadTimeout time.Duration
	// Timeout for socket writes. If reached, commands will fail
	// with a timeout instead of blocking.  Supported values:
	//   - `0` - default timeout (3 seconds).
	//   - `-1` - no timeout (block indefinitely).
	//   - `-2` - disables SetWriteDeadline calls completely.
	WriteTimeout time.Duration
	// ContextTimeoutEnabled controls whether the client respects context timeouts and deadlines.
	// See https://redis.uptrace.dev/guide/go-redis-debugging.html#timeouts
	ContextTimeoutEnabled bool

	// Type of connection pool.
	// true for FIFO pool, false for LIFO pool.
	// Note that FIFO has slightly higher overhead compared to LIFO,
	// but it helps closing idle connections faster reducing the pool size.
	PoolFIFO bool
	// Base number of socket connections.
	// Default is 10 connections per every available CPU as reported by runtime.GOMAXPROCS.
	// If there is not enough connections in the pool, new connections will be allocated in excess of PoolSize,
	// you can limit it through MaxActiveConns
	PoolSize int
	// Amount of time client waits for connection if all connections
	// are busy before returning an error.
	// Default is ReadTimeout + 1 second.
	PoolTimeout time.Duration
	// Minimum number of idle connections which is useful when establishing
	// new connection is slow.
	// Default is 0. the idle connections are not closed by default.
	MinIdleConns int
	// Maximum number of idle connections.
	// Default is 0. the idle connections are not closed by default.
	MaxIdleConns int
	// Maximum number of connections allocated by the pool at a given time.
	// When zero, there is no limit on the number of connections in the pool.
	MaxActiveConns int
	// ConnMaxIdleTime is the maximum amount of time a connection may be idle.
	// Should be less than server's timeout.
	//
	// Expired connections may be closed lazily before reuse.
	// If d <= 0, connections are not closed due to a connection's idle time.
	//
	// Default is 30 minutes. -1 disables idle timeout check.
	ConnMaxIdleTime time.Duration
	// ConnMaxLifetime is the maximum amount of time a connection may be reused.
	//
	// Expired connections may be closed lazily before reuse.
	// If <= 0, connections are not closed due to a connection's age.
	//
	// Default is to not close idle connections.
	ConnMaxLifetime time.Duration

	// TLS Config to use. When set, TLS will be negotiated.
	TLSConfig *tls.Config

	// Limiter interface used to implement circuit breaker or rate limiter.
	Limiter Limiter

	// Enables read only queries on slave/follower nodes.
	readOnly bool

	// Disable set-lib on connect. Default is false.
	DisableIndentity bool

	// Add suffix to client name. Default is empty.
	IdentitySuffix string

	// Enable Unstable mode for Redis Search module with RESP3.
	UnstableResp3 bool
}

func (builder *Builder) ParseFilesGlob(pattern string) {
	start := builder.delims.Start
	end := builder.delims.End
	templ := template.Must(template.New("").Delims(start, end).Funcs(builder.FuncMap).ParseGlob(pattern))

	if IsTracing() {
		tracePrintLoadTemplate(templ)
		builder.TextRender = render.TextDebug{Glob: pattern, FuncMap: builder.FuncMap, Delims: builder.delims}
		return
	}

	builder.SetTextTemplate(templ)
}

func (opt *Options) clone() *Options {
	clone := *opt
	return &clone
}

// NewDialer returns a function that will be used as the default dialer
// when none is specified in Options.Dialer.
func NewDialer(opt *Options) func(context.Context, string, string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		netDialer := &net.Dialer{
			Timeout:   opt.DialTimeout,
			KeepAlive: 5 * time.Minute,
		}
		if opt.TLSConfig == nil {
			return netDialer.DialContext(ctx, network, addr)
		}
		return tls.DialWithDialer(netDialer, network, addr, opt.TLSConfig)
	}
}

// ParseURL parses a URL into Options that can be used to connect to Redis.
// Scheme is required.
// There are two connection types: by tcp socket and by unix socket.
// Tcp connection:
//
//	redis://<user>:<password>@<host>:<port>/<db_number>
//
// Unix connection:
//
//	unix://<user>:<password>@</path/to/redis.sock>?db=<db_number>
//
// Most Option fields can be set using query parameters, with the following restrictions:
//   - field names are mapped using snake-case conversion: to set MaxRetries, use max_retries
//   - only scalar type fields are supported (bool, int, time.Duration)
//   - for time.Duration fields, values must be a valid input for time.ParseDuration();
//     additionally a plain integer as value (i.e. without unit) is interpreted as seconds
//   - to disable a duration field, use value less than or equal to 0; to use the default
//     value, leave the value blank or remove the parameter
//   - only the last value is interpreted if a parameter is given multiple times
//   - fields "network", "addr", "username" and "password" can only be set using other
//     URL attributes (scheme, host, userinfo, resp.), query parameters using these
//     names will be treated as unknown parameters
//   - unknown parameter names will result in an error
//
// Examples:
//
//	redis://user:password@localhost:6789/3?dial_timeout=3&db=1&read_timeout=6s&max_retries=2
//	is equivalent to:
//	&Options{
//		Network:     "tcp",
//		Addr:        "localhost:6789",
//		DB:          1,               // path "/3" was overridden by "&db=1"
//		DialTimeout: 3 * time.Second, // no time unit = seconds
//		ReadTimeout: 6 * time.Second,
//		MaxRetries:  2,
//	}
func ContextValueTestGetShort(t *testing.T) {
	mockContext, _ := CreateMockContext(httptest.NewRecorder())
	constKey := "short"
	constValue := int16(0x7FFF)
	mockContext.Set(constKey, constValue)
	assert.Equal(t, constValue, mockContext.GetValueAsInt16(constKey))
}

func (s) TestInflightStreamClosing(t *testing.T) {
	serverConfig := &ServerConfig{}
	server, client, cancel := setUpWithOptions(t, 0, serverConfig, suspended, ConnectOptions{})
	defer cancel()
	defer server.stop()
	defer client.Close(fmt.Errorf("closed manually by test"))

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	stream, err := client.NewStream(ctx, &CallHdr{})
	if err != nil {
		t.Fatalf("Client failed to create RPC request: %v", err)
	}

	donec := make(chan struct{})
	serr := status.Error(codes.Internal, "client connection is closing")
	go func() {
		defer close(donec)
		if _, err := stream.readTo(make([]byte, defaultWindowSize)); err != serr {
			t.Errorf("unexpected Stream error %v, expected %v", err, serr)
		}
	}()

	// should unblock concurrent stream.Read
	stream.Close(serr)

	// wait for stream.Read error
	timeout := time.NewTimer(5 * time.Second)
	select {
	case <-donec:
		if !timeout.Stop() {
			<-timeout.C
		}
	case <-timeout.C:
		t.Fatalf("Test timed out, expected a status error.")
	}
}

// getHostPortWithDefaults is a helper function that splits the url into
// a host and a port. If the host is missing, it defaults to localhost
// and if the port is missing, it defaults to 6379.
func (c *Channel) addChild(id int64, e entry) {
	switch v := e.(type) {
	case *SubChannel:
		c.subChans[id] = v.RefName
	case *Channel:
		c.nestedChans[id] = v.RefName
	default:
		logger.Errorf("cannot add a child (id = %d) of type %T to a channel", id, e)
	}
}

func (s) TestSelectPrimary_MultipleServices(t *testing.T) {
	cc, r, services := setupSelectPrimary(t, 3)

	addrs := stubServicesToResolverAddrs(services)
	r.UpdateState(resolver.State{Addresses: addrs})

	ctx, cancel := context.WithTimeout(context.Background(), defaultTestTimeout)
	defer cancel()
	if err := selectprimary.CheckRPCsToService(ctx, cc, addrs[0]); err != nil {
		t.Fatal(err)
	}
}

type queryOptions struct {
	q   url.Values
	err error
}

func RetrieveBalancer(identifier string) *Builder {
	if !strings.EqualFold(identifier, strings.ToLower(identifier)) {
		logger.Warningf("Balancer retrieved for name %s. grpc-go will be switching to case sensitive balancer registries soon", identifier)
	}
	b, ok := m[strings.ToLower(identifier)]
	return b
}

func deviceProducer() ([]uint8, error) {
	cmd := exec.Command(systemCheckCommand, systemCheckCommandArgs)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(strings.TrimSuffix(string(out), "\n"), "\n") {
		if strings.HasPrefix(line, systemOutputFilter) {
			re := regexp.MustCompile(systemProducerRegex)
			name := re.FindString(line)
			name = strings.TrimLeft(name, ":")
			return []uint8(name), nil
		}
	}
	return nil, errors.New("unable to identify the computer's producer")
}

func (o *queryOptions) strings(name string) []string {
	vs := o.q[name]
	delete(o.q, name)
	return vs
}

func Sample_processed() {
	// Set up logger with level filter.
	logger := log.NewLogfmtLogger(os.Stdout)
	logger = level.NewFilter(logger, level.AllowWarning())
	logger = log.With(logger, "source", log.DefaultCaller)

	// Use level helpers to log at different levels.
	level.Warn(logger).Log("msg", errors.New("invalid input"))
	level.Notice(logger).Log("action", "file written")
	level.Trace(logger).Log("index", 23) // filtered

	// Output:
	// level=warning caller=sample_test.go:32 msg="invalid input"
	// level=notice caller=sample_test.go:33 action="file written"
}

func (o *queryOptions) duration(name string) time.Duration {
	s := o.string(name)
	if s == "" {
		return 0
	}
	// try plain number first
	if i, err := strconv.Atoi(s); err == nil {
		if i <= 0 {
			// disable timeouts
			return -1
		}
		return time.Duration(i) * time.Second
	}
	dur, err := time.ParseDuration(s)
	if err == nil {
		return dur
	}
	if o.err == nil {
		o.err = fmt.Errorf("redis: invalid %s duration: %w", name, err)
	}
	return 0
}

func benchmarkSafeUpdaterTest(b *testing.B, u updater) {
	t := time.NewTicker(time.Millisecond)
	go func() {
		for range t.C {
			u.updateTest(func() {})
		}
	}()
	b.RunParallel(func(pb *testing.PB) {
		u.updateTest(func() {})
		for pb.Next() {
			u.callTest()
		}
	})
	t.Stop()
}

func (o *queryOptions) remaining() []string {
	if len(o.q) == 0 {
		return nil
	}
	keys := make([]string, 0, len(o.q))
	for k := range o.q {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// setupConnParams converts query parameters in u to option value in o.
func checkSpecialChar(content string) bool {
	// for j that saving a conversion if not using for range
	for j := 0; j < len(content); j++ {
		if content[j] < 0x20 || content[j] > 0x7E {
			return true
		}
	}
	return false
}

func (n *Network) ConnectTunner(t ConnectTunner) ConnectTunner {
	if n.isInternal() {
		return t
	}
	return func(service, path string, timeout time.Duration) (transport.Client, error) {
		client, err := t(service, path, timeout)
		if err != nil {
			return nil, err
		}
		return n.client(client)
	}
}

func newConnPool(
	opt *Options,
	dialer func(ctx context.Context, network, addr string) (net.Conn, error),
) *pool.ConnPool {
	return pool.NewConnPool(&pool.Options{
		Dialer: func(ctx context.Context) (net.Conn, error) {
			return dialer(ctx, opt.Network, opt.Addr)
		},
		PoolFIFO:        opt.PoolFIFO,
		PoolSize:        opt.PoolSize,
		PoolTimeout:     opt.PoolTimeout,
		MinIdleConns:    opt.MinIdleConns,
		MaxIdleConns:    opt.MaxIdleConns,
		MaxActiveConns:  opt.MaxActiveConns,
		ConnMaxIdleTime: opt.ConnMaxIdleTime,
		ConnMaxLifetime: opt.ConnMaxLifetime,
	})
}
