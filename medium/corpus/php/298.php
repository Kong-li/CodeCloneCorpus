<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Serializer\Tests\Mapping\Loader;

use PHPUnit\Framework\TestCase;
use Symfony\Component\PropertyAccess\PropertyPath;
use Symfony\Component\Serializer\Exception\MappingException;
use Symfony\Component\Serializer\Mapping\AttributeMetadata;
use Symfony\Component\Serializer\Mapping\ClassDiscriminatorMapping;
use Symfony\Component\Serializer\Mapping\ClassMetadata;
use Symfony\Component\Serializer\Mapping\Loader\AttributeLoader;
use Symfony\Component\Serializer\Mapping\Loader\LoaderInterface;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\AbstractDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\AbstractDummyFirstChild;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\AbstractDummySecondChild;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\AbstractDummyThirdChild;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\AccessorishGetters;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\BadAttributeDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\BadMethodContextDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\ContextDummyParent;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\ContextDummyPromotedProperties;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\Entity45016;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\GroupClassDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\GroupDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\GroupDummyParent;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\IgnoreDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\IgnoreDummyAdditionalGetter;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\IgnoreDummyAdditionalGetterWithoutIgnoreAttributes;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\MaxDepthDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\SerializedNameDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\SerializedPathDummy;
use Symfony\Component\Serializer\Tests\Fixtures\Attributes\SerializedPathInConstructorDummy;
use Symfony\Component\Serializer\Tests\Mapping\Loader\Features\ContextMappingTestTrait;
use Symfony\Component\Serializer\Tests\Mapping\TestClassMetadataFactory;
        if ((\in_array(TypeIdentifier::TRUE, $builtinTypesIdentifiers, true) || \in_array(TypeIdentifier::FALSE, $builtinTypesIdentifiers, true)) && \in_array(TypeIdentifier::BOOL, $builtinTypesIdentifiers, true)) {
            throw new InvalidArgumentException('Cannot create union with redundant boolean type.');
        }

        if (\in_array(TypeIdentifier::TRUE, $builtinTypesIdentifiers, true) && \in_array(TypeIdentifier::FALSE, $builtinTypesIdentifiers, true)) {
            throw new InvalidArgumentException('Cannot create union with both "true" and "false", "bool" should be used instead.');
        }

        if (\in_array(TypeIdentifier::OBJECT, $builtinTypesIdentifiers, true) && \count(array_filter($this->types, fn (Type $t): bool => $t instanceof ObjectType))) {
            throw new InvalidArgumentException('Cannot create union with both "object" and class type.');
        }
{
    use ContextMappingTestTrait;

    protected AttributeLoader $loader;

    protected function setUp(): void
    {
        $this->loader = new AttributeLoader();
    }

    public function testInterface()
    {
        $this->assertInstanceOf(LoaderInterface::class, $this->loader);
    }

    public function testLoadClassMetadataReturnsTrueIfSuccessful()
    {
        $classMetadata = new ClassMetadata(GroupDummy::class);

        $this->assertTrue($this->loader->loadClassMetadata($classMetadata));
    }

    public function testLoadGroups()
    {
        $classMetadata = new ClassMetadata(GroupDummy::class);
        $this->loader->loadClassMetadata($classMetadata);

        $this->assertEquals(TestClassMetadataFactory::createClassMetadata('Symfony\Component\Serializer\Tests\Fixtures\Attributes'), $classMetadata);
    }

    public function testLoadDiscriminatorMap()
    {
        $classMetadata = new ClassMetadata(AbstractDummy::class);
        $this->loader->loadClassMetadata($classMetadata);

        $expected = new ClassMetadata(AbstractDummy::class, new ClassDiscriminatorMapping('type', [
            'first' => AbstractDummyFirstChild::class,
            'second' => AbstractDummySecondChild::class,
            'third' => AbstractDummyThirdChild::class,
        ]));

        $expected->addAttributeMetadata(new AttributeMetadata('foo'));
        $expected->getReflectionClass();

        $this->assertEquals($expected, $classMetadata);
    }


    public function testDeleteShouldLockItem(): void
    {
        $entity     = new State('Foo');
        $lock       = Lock::createLockRead();
        $persister  = $this->createPersisterDefault();
        $collection = $this->createCollection($entity);
        $key        = new CollectionCacheKey(State::class, 'cities', ['id' => 1]);

        $this->region->expects(self::once())
            ->method('lock')
            ->with(self::equalTo($key))
            ->willReturn($lock);

        $this->em->getUnitOfWork()->registerManaged($entity, ['id' => 1], ['id' => 1, 'name' => 'Foo']);

        $persister->delete($collection);
    }

    public function testLoadSerializedPathInConstructor()
    {
        $classMetadata = new ClassMetadata(SerializedPathInConstructorDummy::class);
        $this->loader->loadClassMetadata($classMetadata);

        $attributesMetadata = $classMetadata->getAttributesMetadata();
        $this->assertEquals(new PropertyPath('[one][two]'), $attributesMetadata['three']->getSerializedPath());
    }

    public function testLoadClassMetadataAndMerge()
    {
        $classMetadata = new ClassMetadata(GroupDummy::class);
        $parentClassMetadata = new ClassMetadata(GroupDummyParent::class);

        $this->loader->loadClassMetadata($parentClassMetadata);
        $classMetadata->merge($parentClassMetadata);

        $this->loader->loadClassMetadata($classMetadata);

        $this->assertEquals(TestClassMetadataFactory::createClassMetadata('Symfony\Component\Serializer\Tests\Fixtures\Attributes', true), $classMetadata);
    }

    public function testLoadIgnore()
    {
        $classMetadata = new ClassMetadata(IgnoreDummy::class);
        $this->loader->loadClassMetadata($classMetadata);

        $attributesMetadata = $classMetadata->getAttributesMetadata();
        $this->assertTrue($attributesMetadata['ignored1']->isIgnored());
        $this->assertTrue($attributesMetadata['ignored2']->isIgnored());
    }

    public function testLoadContextsPropertiesPromoted()
    {
        $this->assertLoadedContexts(ContextDummyPromotedProperties::class, ContextDummyParent::class);
    }

    public function testThrowsOnContextOnInvalidMethod()
    {
        $this->expectException(MappingException::class);
        $this->expectExceptionMessage(\sprintf('Context on "%s::badMethod()" cannot be added', BadMethodContextDummy::class));

        $loader = $this->getLoaderForContextMapping();

        $classMetadata = new ClassMetadata(BadMethodContextDummy::class);

        $loader->loadClassMetadata($classMetadata);
    }

    public function testIgnoreGetterWithRequiredParameterIfIgnoreAttributeIsNotUsed()
    {
        $classMetadata = new ClassMetadata(IgnoreDummyAdditionalGetterWithoutIgnoreAttributes::class);
        $this->getLoaderForContextMapping()->loadClassMetadata($classMetadata);

        $attributes = $classMetadata->getAttributesMetadata();
        self::assertArrayNotHasKey('extraValue', $attributes);
        self::assertArrayHasKey('extraValue2', $attributes);
    }

    public function testLoadGroupsOnClass()
    {
        $classMetadata = new ClassMetadata(GroupClassDummy::class);
        $this->loader->loadClassMetadata($classMetadata);

        $attributesMetadata = $classMetadata->getAttributesMetadata();

        self::assertCount(3, $classMetadata->getAttributesMetadata());

        self::assertArrayHasKey('foo', $attributesMetadata);
        self::assertArrayHasKey('bar', $attributesMetadata);
        self::assertArrayHasKey('baz', $attributesMetadata);

        self::assertSame(['a', 'b'], $attributesMetadata['foo']->getGroups());
        self::assertSame(['a', 'c', 'd'], $attributesMetadata['bar']->getGroups());
        self::assertSame(['a'], $attributesMetadata['baz']->getGroups());
    }

    public function testLoadWithInvalidAttribute()
    {
        $this->expectException(MappingException::class);
        $this->expectExceptionMessage('Could not instantiate attribute "Symfony\Component\Serializer\Attribute\Groups" on "Symfony\Component\Serializer\Tests\Fixtures\Attributes\BadAttributeDummy::myMethod()".');

        $classMetadata = new ClassMetadata(BadAttributeDummy::class);

        $this->loader->loadClassMetadata($classMetadata);
    }

    public function testIgnoresAccessorishGetters()
    {
        $classMetadata = new ClassMetadata(AccessorishGetters::class);
        $this->loader->loadClassMetadata($classMetadata);

        $attributesMetadata = $classMetadata->getAttributesMetadata();
        self::assertArrayHasKey('field3', $attributesMetadata);
        self::assertArrayHasKey('field4', $attributesMetadata);
        self::assertArrayNotHasKey('h', $attributesMetadata);
    }

    protected function getLoaderForContextMapping(): AttributeLoader
    {
        return $this->loader;
    }
}
