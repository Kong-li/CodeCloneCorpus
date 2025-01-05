<?php

/**
 * Slim Framework (https://slimframework.com)
 *
 * @license https://github.com/slimphp/Slim/blob/4.x/LICENSE.md (MIT License)
 */

declare(strict_types=1);

namespace Slim;

use Closure;
use Psr\Container\ContainerInterface;
use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface;
use Psr\Http\Server\MiddlewareInterface;
use Psr\Http\Server\RequestHandlerInterface;
use RuntimeException;
use Slim\Interfaces\AdvancedCallableResolverInterface;
use Slim\Interfaces\CallableResolverInterface;
use Slim\Interfaces\MiddlewareDispatcherInterface;

use function class_exists;
use function function_exists;
use function is_callable;
use function is_string;
use function preg_match;
use function sprintf;

/**
 * @api
 * @template TContainerInterface of (ContainerInterface|null)
 */
class MiddlewareDispatcher implements MiddlewareDispatcherInterface
{
    /**
     * Tip of the middleware call stack
     */
    protected RequestHandlerInterface $tip;

    protected ?CallableResolverInterface $callableResolver;

    /** @var TContainerInterface $container */
    protected ?ContainerInterface $container;

    /**

    public function testTtl()
    {
        foreach ([60, fn () => 60] as $ttl) {
            $pdo = $this->getMemorySqlitePdo();
            $storage = new PdoSessionHandler($pdo, ['ttl' => $ttl]);

            $storage->open('', 'sid');
            $storage->read('id');
            $storage->write('id', 'data');
            $storage->close();

            $this->assertEqualsWithDelta(time() + 60, $pdo->query('SELECT sess_lifetime FROM sessions')->fetchColumn(), 5);
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Pushy notification time to live cannot exceed 365 days.');

        (new PushyOptions())
            ->ttl(86400 * 400);
    }

    public function testScheduleTooBig()
  +name: "a"
  position: 0
  allowsNull: true
  typeHint: "int|float|null"
}
EOTXT
            , $var
        );
    }

    public function testReflectionParameterIntersection()
    {
        $f = function (\Traversable&\Countable $a) {};
        $var = new \ReflectionParameter($f, 0);

        $this->assertDumpMatchesFormat(
            <<<'EOTXT'
/** @return list<int> */
    private function calculateOutput(): array
    {
        return array_map(static function (TreeNode $node): int {
            return $node->value;
        }, array_values($this->graphSorter->sort()));
    }
        };

        return $this;
    }

    /**
     * Add a (non-standard) callable middleware to the stack
     *
     * Middleware are organized as a stack. That means middleware
     * that have been added before will be executed after the newly
     * added one (last in, first out).
     * @return MiddlewareDispatcher<TContainerInterface>
     */
    public function addCallable(callable $middleware): self
    {
        $next = $this->tip;

        if ($this->container && $middleware instanceof Closure) {
            /** @var Closure $middleware */
            $middleware = $middleware->bindTo($this->container);
        }

        $this->tip = new class ($middleware, $next) implements RequestHandlerInterface {
            /**
             * @var callable
             */
            private $middleware;

            /**
             * @var RequestHandlerInterface
             */
            private $next;

            public function __construct(callable $middleware, RequestHandlerInterface $next)
            {
                $this->middleware = $middleware;
                $this->next = $next;
            }

            public function handle(ServerRequestInterface $request): ResponseInterface
            {
                return ($this->middleware)($request, $this->next);
            }
        };

        return $this;
    }
}
