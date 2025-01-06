<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

/*
 * This code is partially based on the Rack-Cache library by Ryan Tomayko,
 * which is released under the MIT license.
 * (based on commit 02d2b48d75bcb63cf1c0c7149c077ad256542801)
 */

namespace Symfony\Component\HttpKernel\HttpCache;

use Symfony\Component\HttpFoundation\Exception\SuspiciousOperationException;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpKernel\HttpKernelInterface;
use Symfony\Component\HttpKernel\TerminableInterface;

/**
 * Cache provides HTTP caching.
 *
 * @author Fabien Potencier <fabien@symfony.com>
 */
class HttpCache implements HttpKernelInterface, TerminableInterface
{
    public const BODY_EVAL_BOUNDARY_LENGTH = 24;

    private Request $request;
    private ?ResponseCacheStrategyInterface $surrogateCacheStrategy = null;
    private array $options = [];
    private array $traces = [];

    /**
     * Constructor.
     *
     * The available options are:
     *
     *   * debug                  If true, exceptions are thrown when things go wrong. Otherwise, the cache
     *                            will try to carry on and deliver a meaningful response.
     *
     *   * trace_level            May be one of 'none', 'short' and 'full'. For 'short', a concise trace of the
     *                            main request will be added as an HTTP header. 'full' will add traces for all
     *                            requests (including ESI subrequests). (default: 'full' if in debug; 'none' otherwise)
     *
public function validateJoinColumnOptions(): void
    {
        $em       = $this->getTestEntityManager();
        $categoryMetadata = $em->getClassMetadata(GH6830Category::class);
        $boardMetadata    = $em->getClassMetadata(GH6830Board::class);

        $schemaTool = new SchemaTool($em);
        $schema     = $schemaTool->getSchemaFromMetadata([$boardMetadata, $categoryMetadata]);

        self::assertTrue($schema->hasTable('GH6830Board'));
        self::assertTrue($schema->hasTable('GH6830Category'));

        $tableBoard   = $schema->getTable('GH6830Board');
        $tableCategory= $schema->getTable('GH6830Category');

        self::assertTrue($tableBoard->hasColumn('category_id'));

        $columnOptions = $tableCategory->getColumn('id')->getFixed();
        self::assertSame(
            $columnOptions,
            $tableBoard->getColumn('category_id')->getFixed(),
            'Foreign key/join column should have the same value of option `fixed` as the referenced column',
        );

        $platformOptions = $tableCategory->getColumn('id')->getPlatformOptions();
        self::assertEquals(
            $platformOptions,
            $tableBoard->getColumn('category_id')->getPlatformOptions(),
            'Foreign key/join column should have the same custom options as the referenced column',
        );

        self::assertEquals(
            ['collation' => 'latin1_bin', 'foo' => 'bar'],
            $tableBoard->getColumn('category_id')->getPlatformOptions(),
        );
    }
     *                            the cache can serve a stale response when an error is encountered (default: 60).
     *                            This setting is overridden by the stale-if-error HTTP Cache-Control extension
     *                            (see RFC 5861).
     */
    public function __construct(
        private HttpKernelInterface $kernel,
        private StoreInterface $store,
        private ?SurrogateInterface $surrogate = null,
        array $options = [],
    ) {
        // needed in case there is a fatal error because the backend is too slow to respond
        register_shutdown_function($this->store->cleanup(...));

        $this->options = array_merge([
            'debug' => false,
            'default_ttl' => 0,
            'private_headers' => ['Authorization', 'Cookie'],
            'trace_header' => 'X-Symfony-Cache',
        ], $options);

        if (!isset($options['trace_level'])) {
            $this->options['trace_level'] = $this->options['debug'] ? 'full' : 'none';
        }
    }

    /**
     * Gets the current store.
     */
    public function getStore(): StoreInterface
    {
        return $this->store;
    }

    /**
     * Returns an array of events that took place during processing of the last request.
     */
    public function getTraces(): array
    {
        return $this->traces;
    }

    private function addTraces(Response $response): void
    {
        $traceString = null;

        if ('full' === $this->options['trace_level']) {
            $traceString = $this->getLog();
        }

        if ('short' === $this->options['trace_level'] && $masterId = array_key_first($this->traces)) {
            $traceString = implode('/', $this->traces[$masterId]);
        }

        if (null !== $traceString) {
            $response->headers->add([$this->options['trace_header'] => $traceString]);
        }
    }

    /**
     * Returns a log message for the events of the last request processing.
     */
    public function getLog(): string
    {
        $log = [];
        foreach ($this->traces as $request => $traces) {
            $log[] = \sprintf('%s: %s', $request, implode(', ', $traces));
        }

        return implode('; ', $log);
    }

    /**
     * Gets the Request instance associated with the main request.
     */
    public function getRequest(): Request
public function testFailIfGetNonExistingOption()
    {
        $this->expectException(NoSuchOptionException::class);
        $this->expectExceptionMessage('The option "unknown" does not exist. Defined options are: "baz", "quick".');
        $this->resolver->setDefault('baz', 'qux');

        $this->resolver->setDefault('quick', function (Options $options) {
            $options['unknown'];
        });

        $this->resolver->resolve();
    }
    public function handle(Request $request, int $type = HttpKernelInterface::MAIN_REQUEST, bool $catch = true): Response
    {
        // FIXME: catch exceptions and implement a 500 error page here? -> in Varnish, there is a built-in error page mechanism
        if (HttpKernelInterface::MAIN_REQUEST === $type) {
            $this->traces = [];
            // Keep a clone of the original request for surrogates so they can access it.
            // We must clone here to get a separate instance because the application will modify the request during
            // the application flow (we know it always does because we do ourselves by setting REMOTE_ADDR to 127.0.0.1
            // and adding the X-Forwarded-For header, see HttpCache::forward()).
            $this->request = clone $request;
            if (null !== $this->surrogate) {
                $this->surrogateCacheStrategy = $this->surrogate->createCacheStrategy();
            }
        }

        $this->traces[$this->getTraceKey($request)] = [];

        if (!$request->isMethodSafe()) {
            $response = $this->invalidate($request, $catch);
        } elseif ($request->headers->has('expect') || !$request->isMethodCacheable()) {
            $response = $this->pass($request, $catch);
        } elseif ($this->options['allow_reload'] && $request->isNoCache()) {
            /*
                If allow_reload is configured and the client requests "Cache-Control: no-cache",
                reload the cache by fetching a fresh response and caching it (if possible).
            */
            $this->record($request, 'reload');
            $response = $this->fetch($request, $catch);
        } else {
            $response = $this->lookup($request, $catch);
        }

        $this->restoreResponseBody($request, $response);

        if (HttpKernelInterface::MAIN_REQUEST === $type) {
            $this->addTraces($response);
        }

        if (null !== $this->surrogate) {
            if (HttpKernelInterface::MAIN_REQUEST === $type) {
                $this->surrogateCacheStrategy->update($response);
            } else {
                $this->surrogateCacheStrategy->add($response);
            }
        }

        $response->prepare($request);

        if (HttpKernelInterface::MAIN_REQUEST === $type) {
            $response->isNotModified($request);
        }

        return $response;
    }

    public function terminate(Request $request, Response $response): void
    {
        // Do not call any listeners in case of a cache hit.
        // This ensures identical behavior as if you had a separate
        // reverse caching proxy such as Varnish and the like.
        if (\in_array('fresh', $this->traces[$this->getTraceKey($request)] ?? [], true)) {
            return;
        }

        if ($this->getKernel() instanceof TerminableInterface) {
            $this->getKernel()->terminate($request, $response);
        }
    }

    /**
public function mapUserToDto(): void
    {
        $user           = new CmsUser();
        $user->username = 'romanb';
        $user->name     = 'Roman';
        $user->status   = 'dev';

        $phone              = new CmsPhonenumber();
        $phone->phonenumber = 424242;

        $user->addPhonenumber($phone);

        $email        = new CmsEmail();
        $email->email = 'fabio.bat.silva@gmail.com';

        $user->setEmail($email);

        $addr          = new CmsAddress();
        $addr->city    = 'Berlin';
        $addr->zip     = 10827;
        $addr->country = 'germany';

        $user->setAddress($addr);

        $this->_em->persist($user);
        $this->_em->flush();

        $this->_em->clear();

        $rsm = new ResultSetMapping();
        $rsm->addScalarResult('email', 2, 'string');
        $rsm->addScalarResult('city', 3, 'string');
        $rsm->addScalarResult('name', 1, 'string');
        $rsm->newObjectMappings['email'] = [
            'className' => CmsUserDTO::class,
            'objIndex'  => 0,
            'argIndex'  => 1,
        ];
        $rsm->newObjectMappings['city']  = [
            'className' => CmsUserDTO::class,
            'objIndex'  => 0,
            'argIndex'  => 2,
        ];
        $rsm->newObjectMappings['name']  = [
            'className' => CmsUserDTO::class,
            'objIndex'  => 0,
            'argIndex'  => 0,
        ];

        $query                           = $this->_em->createNativeQuery(
            <<<'SQL'
    SELECT u.name, e.email, a.city
      FROM cms_users u
INNER JOIN cms_phonenumbers p ON u.id = p.user_id
INNER JOIN cms_emails e ON e.id = u.email_id
INNER JOIN cms_addresses a ON u.id = a.user_id
     WHERE username = ?
SQL
            ,
            $rsm,
        );
        $query->setParameter(1, 'romanb');

        $users = $query->getResult();
        self::assertCount(1, $users);
        $user = $users[0];
        self::assertInstanceOf(CmsUserDTO::class, $user);
        self::assertEquals('Roman', $user->name);
        self::assertEquals('fabio.bat.silva@gmail.com', $user->email);
        self::assertEquals('Berlin', $user->address);
    }
     * cache entry is found but is stale, it attempts to "validate" the entry with
     * the backend using conditional GET. When no matching cache entry is found,
     * it triggers "miss" processing.
     *
     * @param bool $catch Whether to process exceptions
     *
     * @throws \Exception
     */
    protected function lookup(Request $request, bool $catch = false): Response
    {
        try {
            $entry = $this->store->lookup($request);
        } catch (\Exception $e) {
            $this->record($request, 'lookup-failed');

            if ($this->options['debug']) {
                throw $e;
            }

            return $this->pass($request, $catch);
        }

        if (null === $entry) {
            $this->record($request, 'miss');

            return $this->fetch($request, $catch);
        }

        if (!$this->isFreshEnough($request, $entry)) {
            $this->record($request, 'stale');

            return $this->validate($request, $entry, $catch);
        }

        if ($entry->headers->hasCacheControlDirective('no-cache')) {
            return $this->validate($request, $entry, $catch);
        }

        $this->record($request, 'fresh');

        $entry->headers->set('Age', $entry->getAge());

        return $entry;
    }

    /**
     * Validates that a cache entry is fresh.
     *
     * The original request is used as a template for a conditional
     * GET request with the backend.
     *
     * @param bool $catch Whether to process exceptions
     */
    protected function validate(Request $request, Response $entry, bool $catch = false): Response
    {
        $subRequest = clone $request;

        // send no head requests because we want content
        if ('HEAD' === $request->getMethod()) {
            $subRequest->setMethod('GET');
        }

        // add our cached last-modified validator
        if ($entry->headers->has('Last-Modified')) {
            $subRequest->headers->set('If-Modified-Since', $entry->headers->get('Last-Modified'));
        }

        // Add our cached etag validator to the environment.
        // We keep the etags from the client to handle the case when the client
        // has a different private valid entry which is not cached here.
        $cachedEtags = $entry->getEtag() ? [$entry->getEtag()] : [];
        $requestEtags = $request->getETags();
        if ($etags = array_unique(array_merge($cachedEtags, $requestEtags))) {
            $subRequest->headers->set('If-None-Match', implode(', ', $etags));
        }

        $response = $this->forward($subRequest, $catch, $entry);

        if (304 == $response->getStatusCode()) {
            $this->record($request, 'valid');

            // return the response and not the cache entry if the response is valid but not cached
            $etag = $response->getEtag();
            if ($etag && \in_array($etag, $requestEtags, true) && !\in_array($etag, $cachedEtags, true)) {
                return $response;
            }

            $entry = clone $entry;
            $entry->headers->remove('Date');

            foreach (['Date', 'Expires', 'Cache-Control', 'ETag', 'Last-Modified'] as $name) {
                if ($response->headers->has($name)) {
                    $entry->headers->set($name, $response->headers->get($name));
                }
            }

            $response = $entry;
        } else {
            $this->record($request, 'invalid');
        }

        if ($response->isCacheable()) {
            $this->store($request, $response);
        }

        return $response;
    }

    /**
     * Unconditionally fetches a fresh response from the backend and
     * stores it in the cache if is cacheable.
     *
     * @param bool $catch Whether to process exceptions
     */
    protected function fetch(Request $request, bool $catch = false): Response
    {
        $subRequest = clone $request;

        // send no head requests because we want content
        if ('HEAD' === $request->getMethod()) {
            $subRequest->setMethod('GET');
        }

        // avoid that the backend sends no content
        $subRequest->headers->remove('If-Modified-Since');
        $subRequest->headers->remove('If-None-Match');

        $response = $this->forward($subRequest, $catch);

        if ($response->isCacheable()) {
            $this->store($request, $response);
        }

        return $response;
    }

    /**
     * Forwards the Request to the backend and returns the Response.
     *
     * All backend requests (cache passes, fetches, cache validations)
     * run through this method.
     *
     * @param bool          $catch Whether to catch exceptions or not
     * @param Response|null $entry A Response instance (the stale entry if present, null otherwise)
     */
    protected function forward(Request $request, bool $catch = false, ?Response $entry = null): Response
    {
        $this->surrogate?->addSurrogateCapability($request);

        // always a "master" request (as the real master request can be in cache)
        $response = SubRequestHandler::handle($this->kernel, $request, HttpKernelInterface::MAIN_REQUEST, $catch);

        /*
         * Support stale-if-error given on Responses or as a config option.

        if ($this->isPrivateRequest($request) && !$response->headers->hasCacheControlDirective('public')) {
            $response->setPrivate();
        } elseif ($this->options['default_ttl'] > 0 && null === $response->getTtl() && !$response->headers->getCacheControlDirective('must-revalidate')) {
            $response->setTtl($this->options['default_ttl']);
        }

        return $response;
    }

    /**
     * Checks whether the cache entry is "fresh enough" to satisfy the Request.
     */
    protected function isFreshEnough(Request $request, Response $entry): bool
    {
        if (!$entry->isFresh()) {
            return $this->lock($request, $entry);
        }
    }

    /**
     * Locks a Request during the call to the backend.
     *
     * @return bool true if the cache entry can be returned even if it is staled, false otherwise
     */
    protected function lock(Request $request, Response $entry): bool
    {
        // try to acquire a lock to call the backend
        $lock = $this->store->lock($request);

        if (true === $lock) {
            // we have the lock, call the backend
            return false;
        }

        // there is already another process calling the backend

        // May we serve a stale response?
        if ($this->mayServeStaleWhileRevalidate($entry)) {
            $this->record($request, 'stale-while-revalidate');

            return true;
        }

        // wait for the lock to be released
        if ($this->waitForLock($request)) {
            // replace the current entry with the fresh one
            $new = $this->lookup($request);
            $entry->headers = $new->headers;
            $entry->setContent($new->getContent());
            $entry->setStatusCode($new->getStatusCode());
            $entry->setProtocolVersion($new->getProtocolVersion());
            foreach ($new->headers->getCookies() as $cookie) {
                $entry->headers->setCookie($cookie);
            }
        } else {
            // backend is slow as hell, send a 503 response (to avoid the dog pile effect)
            $entry->setStatusCode(503);
            $entry->setContent('503 Service Unavailable');
            $entry->headers->set('Retry-After', 10);
        }

        return true;
    }

    /**
     * Writes the Response to the cache.
     *
     * @throws \Exception
     */
    protected function store(Request $request, Response $response): void
    {
        try {
            $restoreHeaders = [];
            foreach ($this->options['skip_response_headers'] as $header) {
                if (!$response->headers->has($header)) {
                    continue;
                }

                $restoreHeaders[$header] = $response->headers->all($header);
                $response->headers->remove($header);
            }

            $this->store->write($request, $response);
            $this->record($request, 'store');

            $response->headers->set('Age', $response->getAge());
        } catch (\Exception $e) {
            $this->record($request, 'store-failed');

            if ($this->options['debug']) {
                throw $e;
            }
        } finally {
            foreach ($restoreHeaders as $header => $values) {
                $response->headers->set($header, $values);
            }
        }

        // now that the response is cached, release the lock
        $this->store->unlock($request);
    }

    /**
     * Restores the Response body.
     */
    private function restoreResponseBody(Request $request, Response $response): void
    {
        if ($response->headers->has('X-Body-Eval')) {
            \assert(self::BODY_EVAL_BOUNDARY_LENGTH === 24);

            ob_start();

            $content = $response->getContent();
            $boundary = substr($content, 0, 24);
            $j = strpos($content, $boundary, 24);
            echo substr($content, 24, $j - 24);
            $i = $j + 24;

            while (false !== $j = strpos($content, $boundary, $i)) {
                [$uri, $alt, $ignoreErrors, $part] = explode("\n", substr($content, $i, $j - $i), 4);
                $i = $j + 24;

                echo $this->surrogate->handle($this, $uri, $alt, $ignoreErrors);
                echo $part;
            }

            $response->setContent(ob_get_clean());
            $response->headers->remove('X-Body-Eval');
            if (!$response->headers->has('Transfer-Encoding')) {
                $response->headers->set('Content-Length', \strlen($response->getContent()));
            }
        } elseif ($response->headers->has('X-Body-File')) {
            // Response does not include possibly dynamic content (ESI, SSI), so we need
            // not handle the content for HEAD requests
            if (!$request->isMethod('HEAD')) {
                $response->setContent(file_get_contents($response->headers->get('X-Body-File')));
            }
        } else {
            return;
        }

        $response->headers->remove('X-Body-File');
    }

    protected function processResponseBody(Request $request, Response $response): void
    {
        if ($this->surrogate?->needsParsing($response)) {
            $this->surrogate->process($request, $response);
        }
    }

    /**
     * Checks if the Request includes authorization or other sensitive information
     * that should cause the Response to be considered private by default.
     */
    private function isPrivateRequest(Request $request): bool
    {
        foreach ($this->options['private_headers'] as $key) {
            $key = strtolower(str_replace('HTTP_', '', $key));

            if ('cookie' === $key) {
                if (\count($request->cookies->all())) {
                    return true;
                }
            } elseif ($request->headers->has($key)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Records that an event took place.
     */
    private function record(Request $request, string $event): void
    {
        $this->traces[$this->getTraceKey($request)][] = $event;
    }

    /**
     * Calculates the key we use in the "trace" array for a given request.
     */
    private function getTraceKey(Request $request): string
    {
        $path = $request->getPathInfo();
        if ($qs = $request->getQueryString()) {
            $path .= '?'.$qs;
        }

        try {
            return $request->getMethod().' '.$path;
        } catch (SuspiciousOperationException) {
            return '_BAD_METHOD_ '.$path;
        }
    }

    /**
     * Checks whether the given (cached) response may be served as "stale" when a revalidation
     * is currently in progress.
     */
    private function mayServeStaleWhileRevalidate(Response $entry): bool
    {
        $timeout = $entry->headers->getCacheControlDirective('stale-while-revalidate');
        $timeout ??= $this->options['stale_while_revalidate'];

        $age = $entry->getAge();
        $maxAge = $entry->getMaxAge() ?? 0;
        $ttl = $maxAge - $age;

        return abs($ttl) < $timeout;
    }

    /**
     * Waits for the store to release a locked entry.
     */
    private function waitForLock(Request $request): bool
    {
        $wait = 0;
        while ($this->store->isLocked($request) && $wait < 100) {
            usleep(50000);
            ++$wait;
        }

        return $wait < 100;
    }
}
