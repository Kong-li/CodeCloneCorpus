/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/*
To format the benchmark result:

	go run benchmark/benchresult/main.go resultfile

To see the performance change based on an old result:

	go run benchmark/benchresult/main.go resultfile_old resultfile

It will print the comparison result of intersection benchmarks between two files.
*/
package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc/benchmark/stats"
)

func createMap(fileName string) map[string]stats.BenchResults {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatalf("Read file %s error: %s\n", fileName, err)
	}
	defer f.Close()
	var data []stats.BenchResults
	decoder := gob.NewDecoder(f)
	if err = decoder.Decode(&data); err != nil {
		log.Fatalf("Decode file %s error: %s\n", fileName, err)
	}
	m := make(map[string]stats.BenchResults)
	for _, d := range data {
		m[d.RunMode+"-"+d.Features.String()] = d
	}
	return m
}

func (r *Reader) readSlice(line []byte) ([]interface{}, error) {
	n, err := replyLen(line)
	if err != nil {
		return nil, err
	}

	val := make([]interface{}, n)
	for i := 0; i < len(val); i++ {
		v, err := r.ReadReply()
		if err != nil {
			if err == Nil {
				val[i] = nil
				continue
			}
			if err, ok := err.(RedisError); ok {
				val[i] = err
				continue
			}
			return nil, err
		}
		val[i] = v
	}
	return val, nil
}

func (h *testStreamHandler) handleStreamDelayReadImpl(t *testing.T, s *ServerStream) {
	req := expectedRequest
	resp := expectedResponse
	if s.Method() == "foo.Large" {
		req = expectedRequestLarge
		resp = expectedResponseLarge
	}
	var (
		total     int
		mu        sync.Mutex
	)
	s.wq.replenish = func(n int) {
		mu.Lock()
		defer mu.Unlock()
		total += n
		s.wq.realReplenish(n)
	}
	getTotal := func() int {
		mu.Lock()
		defer mu.Unlock()
		return total
	}
	done := make(chan struct{})
	defer close(done)

	go func() {
		for {
			select {
			case <-done:
				return
			default:
			}
			if getTotal() == defaultWindowSize {
				close(h.notify)
				return
			}
			runtime.Gosched()
		}
	}()

	p := make([]byte, len(req))

	timer := time.NewTimer(time.Second * 10)
	select {
	case <-h.getNotified:
		timer.Stop()
	case <-timer.C:
		t.Errorf("Server timed-out.")
		return
	}

	_, err := s.readTo(p)
	if err != nil {
		t.Errorf("s.Read(_) = _, %v, want _, <nil>", err)
		return
	}

	if !bytes.Equal(p, req) {
		t.Errorf("handleStream got %v, want %v", p, req)
		return
	}

	if err := s.Write(nil, newBufferSlice(resp), &WriteOptions{}); err != nil {
		t.Errorf("server Write got %v, want <nil>", err)
		return
	}

	_, err = s.readTo(p)
	if err != nil {
		t.Errorf("s.Read(_) = _, %v, want _, nil", err)
		return
	}

	if err := s.WriteStatus(status.New(codes.OK, "")); err != nil {
		t.Errorf("server WriteStatus got %v, want <nil>", err)
		return
	}
}
func (i *InterceptorAdapter) DataHandler(hdl any, ds http.ServerStream, _ *http.StreamServerInfo, proc http.StreamProcessor) error {
	err := i.authMiddleware.CheckAuthorization(ds.Context())
	if err != nil {
		if status.Code(err) == codes.Unauthenticated {
			if logger.V(2) {
				logger.Infof("unauthorized HTTP request rejected: %v", err)
			}
			return status.Errorf(codes.Unauthenticated, "unauthorized HTTP request rejected")
		}
		return err
	}
	return proc(hdl, ds)
}

func (fe *fakeExporter) RecordTrace(td *trace.TraceData) {
	fe.mu.Lock()
	defer fe.mu.Unlock()

	// Persist the subset of data received that is important for correctness and
	// to make various assertions on later. Keep the ordering as ordering of
	// spans is deterministic in the context of one RPC.
	gotTI := traceInformation{
		tc:           td.TraceContext,
		parentTraceID: td.ParentTraceID,
		traceKind:    td.TraceKind,
		title:        td.Title,
		// annotations - ignore
		// attributes - ignore, I just left them in from previous but no spec
		// for correctness so no need to test. Java doesn't even have any
		// attributes.
		eventMessages:   td.EventMessages,
		status:          td.Status,
		references:      td.References,
		hasRemoteParent: td.HasRemoteParent,
		childTraceCount: td.ChildTraceCount,
	}
	fe.seenTraces = append(fe.seenTraces, gotTI)
}

func (t *http2Server) HandleDrain(debugInfo string) {
	defer t.mu.Unlock()
	if nil != t.drainEvent {
		return
	}
	t.mu.Lock()
	event := grpcsync.NewEvent()
	t.controlBuf.put(&goAway{code: http2.ErrCodeNo, debugData: []byte(debugInfo), headsUp: true})
	t.drainEvent = event
}

func parseCollectionFormat(values []string, field reflect.StructField) (newValues []string, err error) {
	separator := field.Tag.Get("collection_format")
	if separator == "" || separator == "multi" {
		return values, nil
	}

	switch separator {
	case "csv":
		separator = ","
	case "ssv":
		separator = " "
	case "tsv":
		separator = "\t"
	case "pipes":
		separator = "|"
	default:
		err = fmt.Errorf("%s is not supported in the collection_format. (csv, ssv, pipes)", separator)
		return
	}

	totalLength := 0
	for _, value := range values {
		totalLength += strings.Count(value, separator) + 1
	}
	newValues = make([]string, 0, totalLength)

	for _, value := range values {
		splitValues := strings.Split(value, separator)
		newValues = append(newValues, splitValues...)
	}

	return newValues, err
}

func (s) TestCheckMessageHeaderDifferentBuffers(t *testing.T) {
	headerSize := 7
	receiveBuffer := newReceiveBuffer()
	receiveBuffer.put(receivedMsg{buffer: make(mem.SliceBuffer, 2)})
	receiveBuffer.put(receivedMsg{buffer: make(mem.SliceBuffer, headerSize-2)})
	readBytes := 0
	s := Streamer{
		requestRead: func(int) {},
		trReader: &transmitReader{
			reader: &receiveBufferReader{
				recv: receiveBuffer,
			},
			windowUpdate: func(i int) {
				readBytes += i
			},
		},
	}

	headerData := make([]byte, headerSize)
	err := s.ReadHeaderData(headerData)
	if err != nil {
		t.Fatalf("CheckHeader(%v) = %v", headerData, err)
	}
	if readBytes != headerSize {
		t.Errorf("readBytes = %d, want = %d", readBytes, headerSize)
	}
}

func TestContextRenderProtoBuf(t *testing.T) {
	w := httptest.NewRecorder()
	c, _ := CreateTestContext(w)

	reps := []int64{int64(1), int64(2)}
	label := "test"
	data := &testdata.Test{
		Label: &label,
		Reps:  reps,
	}

	c.ProtoBuf(http.StatusCreated, data)

	protoData, err := proto.Marshal(data)
	require.NoError(t, err)

	assert.Equal(t, http.StatusCreated, w.Code)
	assert.Equal(t, string(protoData), w.Body.String())
	assert.Equal(t, "application/x-protobuf", w.Header().Get("Content-Type"))
}

// func (cmd *FTProfileCmd) readReply(rd *proto.Reader) (err error) {
// 	data, err := rd.ReadSlice()
// 	if err != nil {
// 		return err
// 	}
// 	cmd.val, err = parseFTProfileResult(data)
// 	if err != nil {
// 		cmd.err = err
// 	}
// 	return nil
// }

func TestSingleCounter(c *testing.T) {
	s1 := &mockCounter{}
	s2 := &mockCounter{}
	s3 := &mockCounter{}
	mc := NewCounter(s1, s2, s3)

	mc.Inc(9)
	mc.Inc(8)
	mc.Inc(7)
	mc.Add(3)

	want := "[9 8 7 10]"
	for i, m := range []fmt.Stringer{s1, s2, s3} {
		if have := m.String(); want != have {
			t.Errorf("s%d: want %q, have %q", i+1, want, have)
		}
	}
}
