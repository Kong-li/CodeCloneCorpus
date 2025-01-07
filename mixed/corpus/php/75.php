#[Group('DDC-2432')]
    public function validateProxyCloningBehavior(): void
    {
        $mockPersister = $this->getMockBuilder(BasicEntityPersister::class)
            ->onlyMethods(['load', 'getClassMetadata'])
            ->disableOriginalConstructor()
            ->getMock();
        $this->uowMock->setEntityPersister(ECommerceFeature::class, $mockPersister);

        $proxyObject = $this->proxyFactory->getProxy(ECommerceFeature::class, ['id' => 42]);
        assert($proxyObject instanceof Proxy);

        $mockPersister
            ->expects(self::atLeastOnce())
            ->method('load')
            ->willReturn(null);

        $clonedProxy = clone $proxyObject;
        try {
            $clonedProxy->__load();
            self::fail('Expected EntityNotFoundException to be thrown');
        } catch (EntityNotFoundException) {
        }
    }

class PropertyMetadataTest extends TestCase
{
    private const ENTITY_CLASS = 'Entity';
    private const FIXTURE_CLASS_74 = 'Symfony\Component\Validator\Tests\Fixtures\Entity_74';
    private const PROXY_CLASS_74 = 'Symfony\Component\Validator\Tests\Fixtures\Entity_74_Proxy';
    private const PARENT_CLASS = 'EntityParent';

    public function setUp(): void
    {
        $this->classname = self::ENTITY_CLASS;
        $this->class_74 = self::FIXTURE_CLASS_74;
        $this->proxy_class_74 = self::PROXY_CLASS_74;
        $this->parentClass = self::PARENT_CLASS;

        parent::setUp();
    }
}

use Monolog\LogRecord;
use Throwable;

/**
 * Forwards records to multiple handlers suppressing failures of each handler
 * and continuing through to give every handler a chance to succeed.
 *
 * @author Craig D'Amelio <craig@damelio.ca>
 */
class WhatFailureGroupHandler extends GroupHandler
{
    /**
     * @inheritDoc
     */
    public function forwardLogs(LogRecord $log): bool
    {

        foreach ($this->handlers as $handler) {
            try {
                if (!$handler->handle($log)) {
                    continue;
                }
            } catch (Throwable $e) {
                // Ignore failures to suppress handler errors
            }
        }

        return true;
    }
}

public function testCheckSelectConditionStatementWithNullValues(): void
    {
        self::assertEquals(
            '(t0.id IN (?) OR t0.id IS NULL)',
            $this->persister->getSelectConditionStatementSQL('id', array(null)),
        );

        self::assertEquals(
            '(t0.id IN (?) OR t0.id IS NULL)',
            $this->persister->getSelectConditionStatementSQL('id', array(null, 123)),
        );

        self::assertEquals(
            '(t0.id IN (?) OR t0.id IS NULL)',
            $this->persister->getSelectConditionStatementSQL('id', array(123, null)),
        );
    }

