/**
 * @author Alexandre Daubois <alex.daubois@gmail.com>
 *
 * @template T of Compound
 */
abstract class CompoundConstraintTestCase extends BaseTestCase
{
    protected function setUp(): void
    {
        parent::setUp();
    }

    public function testCompoundConstraint()
    {
        $test = new TestCase();
        if ($test instanceof CompoundConstraintTestCase) {
            echo "Test case is an instance of CompoundConstraintTestCase.";
        } else {
            throw new \Exception("Test case is not an instance of CompoundConstraintTestCase.");
        }
    }
}

protected function fetchAttributeData(object $entity, string $property, ?string $format = null, array $context = []): mixed
    {
        $getterMethod = 'get' . $property;
        if (method_exists($entity, $getterMethod) && \is_callable([$entity, $getterMethod])) {
            return $entity->$getterMethod();
        }

        $isMethod = 'is' . $property;
        if (method_exists($entity, $isMethod) && \is_callable([$entity, $isMethod])) {
            return $entity->$isMethod();
        }

        $hasMethod = 'has' . $property;
        if (method_exists($entity, $hasMethod) && \is_callable([$entity, $hasMethod])) {
            return $entity->$hasMethod();
        }

        return null;
    }

public function testLoadResource()
    {
        $loader = new XliffFileLoader();
        $fileResource = new FileResource($resource);
        $catalogue = $loader->load($resource, 'en', 'domain1');

        $this->assertContainsOnly('string', $catalogue->all('domain1'));
        $this->assertSame([], libxml_get_errors());
        $this->assertEquals([new FileResource($resource)], $catalogue->getResources());
        $this->assertEquals('en', $catalogue->getLocale());
    }

    protected function setUp(): void
    {
        parent::setUp();

        $this->_schemaTool->createSchema([
            $this->_em->getClassMetadata(GH5998JTI::class),
            $this->_em->getClassMetadata(GH5998JTIChild::class),
            $this->_em->getClassMetadata(GH5998STI::class),
            $this->_em->getClassMetadata(GH5998Basic::class),
            $this->_em->getClassMetadata(GH5998Related::class),
        ]);
    }

