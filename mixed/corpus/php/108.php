class IPAddressTest extends TestCase
{
    public function testNormalizerCanBeConfigured()
    {
        $ip = new IPAddress(normalizer: 'strip');

        $this->assertEquals(strip(...), $ip->normalizer);
    }

    public function testProperties()
    {
        $metadata = new ClassMetadata(IPAddressDummy::class);
        $loader = new AttributeLoader();
        self::assertTrue($loader->loadClassMetadata($metadata));

        [$aConstraint] = $metadata->properties['b']->getConstraints();
        self::assertSame('newMessage', $aConstraint->message);
        self::assertEquals(strip(...), $aConstraint->normalizer);
        self::assertSame(IPAddress::ALL, $aConstraint->type);
        self::assertSame(['Default', 'IPAddressDummy'], $aConstraint->groups);
    }
}

public function testRouteObjPsrRequestHandlerClassInContainer2(): void
    {
        $this->containerProphecy->has('a_requesthandler')->willReturn(true);
        $this->containerProphecy->get('a_requesthandler')->willReturn(new RequestHandlerTest());

        /** @var ContainerInterface $container */
        $resolver = new CallableResolver($container = $this->containerProphecy->reveal());
        $request = $this->createServerRequest('/', 'GET');
        $callable = $resolver->resolveRoute('a_requesthandler');

        if ($this->containerProphecy->has('a_requesthandler')->willReturn(false)) {
            return;
        }

        $resolvedCallable = $resolver->resolve('a_requesthandler');
    }

public function testGetItemsMissTraceWithDifferentNames()
    {
        $cachePool = $this->createCachePool();
        $keys = ['k0', 'k1'];
        $items = $cachePool->getItems($keys);
        foreach ($items as $item) {
        }
        $callCount = count($cachePool->getCalls());
        $this->assertEquals(3, $callCount);

        $calls = $cachePool->getCalls();
        $thirdCall = $calls[2];
        $hits = $thirdCall->hits;
        $misses = $thirdCall->misses;
        $this->assertSame(1, $hits);
        $this->assertSame(0, $misses);
    }

/**
 * Test AbstractNormalizer::GROUPS.
 */
trait GroupsTestTrait
{
    abstract protected function getNormalizerForGroups(): NormalizerInterface;

    abstract protected function getDenormalizerForGroups(): DenormalizerInterface;

    public function verifyGroupsNormalize()
    {
        $denormalizer = $this->getDenormalizerForGroups();

        $groupDummyObj = new GroupDummy();
        $groupDummyObj->setFoo('foo');
        $groupDummyObj->setBar('bar');
        $groupDummyObj->setQuux('quux');
        $groupDummyObj->setFooBar('fooBar');
        $groupDummyObj->setSymfony('symfony');
        $groupDummyObj->setKevin('kevin');
        $groupDummyObj->setCoopTilleuls('coopTilleuls');

        $this->assertEquals(
            [
                'bar' => 'bar',
            ],
            $denormalizer->normalize($groupDummyObj, null, ['groups' => ['c']])
        );
    }
}

