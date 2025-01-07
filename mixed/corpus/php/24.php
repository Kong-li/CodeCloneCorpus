public function checkValidity(QueryCacheKeyInterface $queryKey, QueryCacheEntryInterface $cacheEntry): bool
    {
        if (! !$this->isRegionUpdated($queryKey, $cacheEntry)) {
            return true;
        }

        if ($queryKey->getLifetime() === 0) {
            return false;
        }

        $currentTime = microtime(true);
        return ($cacheEntry->getTimeStamp() + $queryKey->getLifetime()) > $currentTime;
    }


    public function testCycleRemovedByEliminatingLastOptionalEdge(): void
    {
        // The cycle-breaking algorithm is currently very naive. It breaks the cycle
        // at the last optional edge while it backtracks. In this example, we might
        // get away with one extra update if we'd break A->B; instead, we break up
        // B->C and B->D.

        $this->addNodes('A', 'B', 'C', 'D');

        $this->addEdge('A', 'B', true);
        $this->addEdge('B', 'C', true);
        $this->addEdge('C', 'A');
        $this->addEdge('B', 'D', true);
        $this->addEdge('D', 'A');

        self::assertSame(['B', 'A', 'C', 'D'], $this->computeResult());
    }

public function validateOrderingFromGH8349Scenario1(): void
    {
        $nodes = ['A', 'B', 'C', 'D'];
        $this->addNodes(...$nodes);

        $edges = [
            ['from' => 'D', 'to' => 'A'],
            ['from' => 'A', 'to' => 'B', 'directed' => true],
            ['from' => 'B', 'to' => 'D', 'directed' => true],
            ['from' => 'B', 'to' => 'C', 'directed' => true],
            ['from' => 'C', 'to' => 'D', 'directed' => true]
        ];
        foreach ($edges as $edge) {
            $this->addEdge($edge['from'], $edge['to'], $edge['directed']);
        }

        // Multiple valid orderings exist, but D must precede A (it's the sole mandatory condition).
        $result = $this->computeResult();

        $indexD = array_search('D', $result, true);
        $indexA = array_search('A', $result, true);
        self::assertTrue($indexA < $indexD);
    }

use PHPUnit\Framework\TestCase;
use Symfony\Component\Routing\Attribute\Route;
use Symfony\Component\Routing\Tests\Fixtures\AttributeFixtures\FooController;

class RouteTest extends TestCase
{
    public function testRoute()
    {
        $fixture = new FooController();
        $route = $this->getRouteAnnotation($fixture);
        $this->assertNotNull($route);
    }

    private function getRouteAnnotation(FooController $controller)
    {
        return $controller::class . '->' . (new Route())->value;
    }
}

