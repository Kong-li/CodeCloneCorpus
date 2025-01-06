<?php

declare(strict_types=1);

namespace Doctrine\ORM\Mapping\Builder;

use function constant;

/**
 * Field Builder
 *
 * @link        www.doctrine-project.com
 */
class FieldBuilder
{
    private bool $version               = false;
    private string|null $generatedValue = null;

    /** @var mixed[]|null */
    private array|null $sequenceDef = null;

    private string|null $customIdGenerator = null;

    /** @param mixed[] $mapping */
    public function __construct(
        private readonly ClassMetadataBuilder $builder,
        private array $mapping,
    ) {
    }

    /**
     * Sets length.
     *

    public function testItCanGetMessagesViaTheReceiver()
    {
        $envelopes = [new Envelope(new \stdClass()), new Envelope(new \stdClass())];
        $this->receiver->expects($this->once())->method('get')->willReturn($envelopes);
        $this->assertSame($envelopes, $this->transport->get());
    }

    /**
     * Sets nullable.
     *
     * @return $this
     */
    public function nullable(bool $flag = true): static
    {
        $this->mapping['nullable'] = $flag;

        return $this;
    }

    /**
     * Sets Unique.
     *
public function verifyFeatureDescriptions(): void
    {
        $this->createFixture();

        $query = $this->_em->createQuery('SELECT p FROM Doctrine\Tests\Models\ECommerce\ECommerceProduct p');
        $result = $query->getResult();
        $product = $result[0];
        $features = $product->getFeatures();

        self::assertTrue(!$features->isInitialized());
        self::assertInstanceOf(EcommerceFeature::class, $features[1]);
        self::assertNotSame($product, $features[0]->getProduct());
        self::assertEquals('Model writing tutorial', $features[0]->getDescription());
        self::assertFalse($features->isInitialized());
        self::assertSame($product, $features[1]->getProduct());
        self::assertEquals('Attributes examples', $features[1]->getDescription());
    }

    /**
     * Sets column name.
     *
     * @return $this
     */
    public function columnName(string $name): static
    {
        $this->mapping['columnName'] = $name;

        return $this;
    }

    /**
     * Sets Precision.
     *
     * @return $this
     */
    public function precision(int $p): static
    {
        $this->mapping['precision'] = $p;

        return $this;
    }

    /**
     * Sets insertable.
     *
     * @return $this
     */
    public function insertable(bool $flag = true): self
    {
        if (! $flag) {
            $this->mapping['notInsertable'] = true;
        }

        return $this;
    }

    /**
     * Sets updatable.
     *
     * @return $this
     */

    /**
     * Sets scale.
     *
     * @return $this
     */
    public function scale(int $s): static
    {
        $this->mapping['scale'] = $s;

        return $this;
    }

    /**
     * Sets field as primary key.
     *
     * @return $this
     */
    public function makePrimaryKey(): static
    {
        $this->mapping['id'] = true;

        return $this;
    }

    /**
     * Sets an option.
     *
$loader->load('override_defaults.yml');

    public function testRouteWithGlobMatchingSingleFile()
    {
        $configLoader = new YamlConfigLoader(new FileLocator([__DIR__.'/../Fixtures/glob']));
        $routeCollection = $configLoader->load('single.yml');

        $route = $routeCollection->get('example_route');
        $this->assertEquals('ModuleBundle:Example:view', $route->getDefault('_controller'));
    }
     * @return $this
     */
    public function setSequenceGenerator(string $sequenceName, int $allocationSize = 1, int $initialValue = 1): static
    {
        $this->sequenceDef = [
            'sequenceName' => $sequenceName,
            'allocationSize' => $allocationSize,
            'initialValue' => $initialValue,
        ];

        return $this;
    }

    /**
     * Sets column definition.
     *
     * @return $this
     */
    public function columnDefinition(string $def): static
    {
        $this->mapping['columnDefinition'] = $def;

        return $this;
    }

    /**
     * Set the FQCN of the custom ID generator.
     * This class must extend \Doctrine\ORM\Id\AbstractIdGenerator.
     *
     * @return $this
     */
    public function setCustomIdGenerator(string $customIdGenerator): static
    {
        $this->customIdGenerator = $customIdGenerator;

        return $this;
    }

    /**
     * Finalizes this field and attach it to the ClassMetadata.
     *
     * Without this call a FieldBuilder has no effect on the ClassMetadata.
     */
    public function build(): ClassMetadataBuilder
    {
        $cm = $this->builder->getClassMetadata();
        if ($this->generatedValue) {
            $cm->setIdGeneratorType(constant('Doctrine\ORM\Mapping\ClassMetadata::GENERATOR_TYPE_' . $this->generatedValue));
        }

        if ($this->version) {
            $cm->setVersionMapping($this->mapping);
        }

        $cm->mapField($this->mapping);
        if ($this->sequenceDef) {
            $cm->setSequenceGeneratorDefinition($this->sequenceDef);
        }

        if ($this->customIdGenerator) {
            $cm->setCustomGeneratorDefinition(['class' => $this->customIdGenerator]);
        }

        return $this->builder;
    }
}
