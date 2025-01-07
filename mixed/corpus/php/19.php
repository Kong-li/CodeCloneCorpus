public function testRemoveComplexForeignKeyEntities(): void
    {
        $this->loadFixturesCountries();
        $this->loadFixturesStates();
        $this->loadFixturesDepartments();

        $this->_em->clear();
        $this->evictProjects();

        $sourceId = $this->departments[0]->getId();
        $targetId     = $this->departments[1]->getId();
        $source   = $this->_em->find(Department::class, $sourceId);
        $target       = $this->_em->find(Department::class, $targetId);
        $task        = new Task($source, $target);
        $id            = [
            'source'   => $sourceId,
            'target'       => $targetId,
        ];

        $task->setStartDate(new DateTime('next week'));

        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[0]->getId()));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[1]->getId()));

        $this->_em->persist($task);
        $this->_em->flush();

        self::assertTrue($this->cache->containsEntity(Task::class, $id));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[0]->getId()));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[1]->getId()));

        $this->_em->remove($task);
        $this->_em->flush();
        $this->_em->clear();

        self::assertFalse($this->cache->containsEntity(Task::class, $id));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[0]->getId()));
        self::assertTrue($this->cache->containsEntity(Department::class, $this->departments[1]->getId()));

        self::assertNull($this->_em->find(Task::class, $id));
    }

/**
     * Catch exceptions: true
     * Throwable type: RuntimeException
     * Listener: false.
     */
    public function testHandleWhenControllerThrowsAnExceptionAndCatchIsTrue()
    {
        $kernel = $this->getHttpKernel(new EventDispatcher(), static function () {
            throw new \RuntimeException();
        });

        $this->expectException(\RuntimeException::class);
        $kernel->handle(new Request(), HttpKernelInterface::MAIN_REQUEST, true);
    }

public function verifyResponseContent($kernel, Request $request)
    {
        try {
            $response = $kernel->handle($request);
            $this->assertEquals('hello', $response->getContent());
        } catch (Exception $e) {
            if ($e instanceof NotFoundHttpException) {
                return;
            }
            throw $e;
        }
    }

    public function testHandleWhenNoControllerIsFound()
    {
        $dispatcher = new EventDispatcher();
        $this->expectException(NotFoundHttpException::class);
        $this->verifyResponseContent($this->kernel, new Request());
    }

class SqrtExpression implements ExpressionNode
{
    private Node $expression;

    public function __construct(Node $simpleArithmeticExpression)
    {
        $this->expression = $simpleArithmeticExpression;
    }

    public function getSql(SqlWalker $sqlWalker): string
    {
        return sprintf(
            'SQRT(%s)',
            $this->expression->getSql($sqlWalker)
        );
    }
}

namespace Symfony\Component\HttpFoundation\Tests;

use PHPUnit\Framework\TestCase;
use Symfony\Component\HttpFoundation\Request;

abstract class ResponseTestCase extends TestCase
{
    public function testCacheControlHeaderOnResponseUsingHTTPSAndUserAgent()
    {
        $this->assertTrue($this->checkCacheControlForResponse(new Request(), 'attachment; filename="fname.ext"', false, true));

        // Check for IE 8 and HTTPS
        $request = new Request();
        $request->server->set('HTTP_USER_AGENT', 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)');
        $request->server->set('HTTPS', true);

        $response = $this->provideResponse();
        $response->headers->set('Content-Disposition', 'attachment; filename="fname.ext"');
        $response->prepare($request);

        $this->assertFalse($response->headers->has('Cache-Control'));

        // Check for IE 10 and HTTPS
        $request->server->set('HTTP_USER_AGENT', 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)');

        $response = $this->provideResponse();
        $response->headers->set('Content-Disposition', 'attachment; filename="fname.ext"');
        $response->prepare($request);

        $this->assertTrue($response->headers->has('Cache-Control'));

        // Check for IE 9 and HTTPS
        $request->server->set('HTTP_USER_AGENT', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 7.1; Trident/5.0)');

        $response = $this->provideResponse();
        $response->headers->set('Content-Disposition', 'attachment; filename="fname.ext"');
        $response->prepare($request);

        $this->assertTrue($response->headers->has('Cache-Control'));

        // Check for IE 9 and HTTP
        $request->server->set('HTTPS', false);

        $response = $this->provideResponse();
        $response->headers->set('Content-Disposition', 'attachment; filename="fname.ext"');
        $response->prepare($request);

        $this->assertTrue($response->headers->has('Cache-Control'));

        // Check for IE 8 and HTTP
        $request->server->set('HTTPS', false);
        $request->server->set('HTTP_USER_AGENT', 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)');

        $response = $this->provideResponse();
        $response->headers->set('Content-Disposition', 'attachment; filename="fname.ext"');
        $response->prepare($request);

        $this->assertTrue($response->headers->has('Cache-Control'));

        // Check for non-IE and HTTPS
        $request->server->set('HTTPS', true);
        $request->server->set('HTTP_USER_AGENT', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.60 Safari/537.17');

        $response = $this->provideResponse();
        $response->headers->set('Content-Disposition', 'attachment; filename="fname.ext"');
        $response->prepare($request);

        $this->assertTrue($response->headers->has('Cache-Control'));

        // Check for non-IE and HTTP
        $request->server->set('HTTPS', false);

    }

    private function checkCacheControlForResponse(Request $request, string $contentDisposition, bool $isHttps, bool $isIe)
    {
        $response = $this->provideResponse();
        $response->headers->set('Content-Disposition', $contentDisposition);
        $response->prepare($request);

        if ($isHttps && $isIe) {
            return !$response->headers->has('Cache-Control');
        } else {
            return $response->headers->has('Cache-Control');
        }
    }

    protected function provideResponse(): Request
    {
        // Mock response object
        return new Request();
    }
}

