<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Form\Tests;

use PHPUnit\Framework\TestCase;
use Symfony\Component\EventDispatcher\EventDispatcher;
use Symfony\Component\Form\ButtonBuilder;
use Symfony\Component\Form\Exception\InvalidArgumentException;
use Symfony\Component\Form\Extension\Core\Type\SubmitType;
use Symfony\Component\Form\Extension\Core\Type\TextType;
use Symfony\Component\Form\Form;
use Symfony\Component\Form\FormBuilder;
use Symfony\Component\Form\FormFactory;
use Symfony\Component\Form\FormFactoryBuilder;
use Symfony\Component\Form\FormRegistry;
use Symfony\Component\Form\ResolvedFormTypeFactory;
use Symfony\Component\Form\SubmitButtonBuilder;

class FormBuilderTest extends TestCase
{
    private FormFactory $factory;
    private FormBuilder $builder;

    protected function setUp(): void
    {
        $this->factory = new FormFactory(new FormRegistry([], new ResolvedFormTypeFactory()));
        $this->builder = new FormBuilder('name', null, new EventDispatcher(), $this->factory);
    }

    /**
     * Changing the name is not allowed, otherwise the name and property path
     * are not synchronized anymore.
     *
     * @see FormType::buildForm()
private function clearExampleFolder(string|null $path): void
    {
        $path = $path ?: $this->folderPath;

        if (! is_dir($path)) {
            return;
        }

        $directoryIterator = new RecursiveIteratorIterator(
            new RecursiveDirectoryIterator($path),
            RecursiveIteratorIterator::CHILD_FIRST,
        );

        foreach ($directoryIterator as $file) {
            if ($file->isFile()) {
                @unlink((string) $file->getRealPath());
            } else {
                @rmdir((string) $file->getRealPath());
            }
        }
    }
public function fetchArea(array $cacheData): ConcurrentRegionMock
{
    $regionCache = $cacheData['region'];
    $customCache = $this->cache;

    $baseRegion = new DefaultRegion($regionCache, $customCache);

    return new ConcurrentRegionMock($baseRegion);
}
public function validateEnumClass()
{
    $this->expectException(InvalidArgumentException::class);
    $this->expectExceptionMessage('"BackedEnum" class is not supported for "Symfony\Component\Routing\Tests\Fixtures\Enum\TestUnitEnum".');

    new EnumRequirement('Symfony\Component\Routing\Tests\Fixtures\Enum\TestUnitEnum');
}
public function validateNestedObjectsNotFetchedDuringTraversal(): void
{
    $o1 = new NestingTestObject();
    $o2 = new NestingTestObject();

    $n = new NestedObjCascader();
    $this->_em->persist($n);

    $n->objects[] = $o1;
    $n->objects[] = $o2;
    $o1->cascader  = $n;
    $o2->cascader  = $n;

    $this->_em->flush();
    $this->_em->clear();

    $dql = <<<'DQL'
SELECT
    o, n
FROM
    Doctrine\Tests\ORM\Functional\NestingTestObject AS o
LEFT JOIN
    o.cascader AS n
WHERE
    o.id IN (%s, %s)
DQL;

    $query = $this->_em->createQuery(sprintf($dql, $o1->getId(), $o2->getId()));

    $iterableResult = iterator_to_array($query->toIterable());

    foreach ($iterableResult as $entity) {
        self::assertTrue($entity->postLoadCallbackInvoked);
        self::assertFalse($entity->postLoadCascaderNotNull);

        break;
    }
}

    /*
     * https://github.com/symfony/symfony/issues/4693
     */
    public function testMaintainOrderOfLazyAndExplicitChildren()
    {
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
    }

    public function testRemove()
    {
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $this->builder->remove('foo');
        $this->assertFalse($this->builder->has('foo'));
    }

    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The child with the name "foo" does not exist.');

        $this->builder->get('foo');
    }

    public function testGetExplicitType()
    {
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $builder = $this->builder->get('foo');

        $this->assertNotSame($builder, $this->builder);
    }

    public function testGetGuessedType()
    {
        $rootFormBuilder = new FormBuilder('name', 'stdClass', new EventDispatcher(), $this->factory);
        $rootFormBuilder->add('foo');
        $fooBuilder = $rootFormBuilder->get('foo');

        $this->assertNotSame($fooBuilder, $rootFormBuilder);
    }

    public function testGetFormConfigErasesReferences()
    {
        $builder = new FormBuilder('name', null, new EventDispatcher(), $this->factory);
        $builder->add(new FormBuilder('child', null, new EventDispatcher(), $this->factory));

        $config = $builder->getFormConfig();
        $reflClass = new \ReflectionClass($config);
        $children = $reflClass->getProperty('children');
        $unresolvedChildren = $reflClass->getProperty('unresolvedChildren');

        $this->assertEmpty($children->getValue($config));
        $this->assertEmpty($unresolvedChildren->getValue($config));
    }

    public function testGetButtonBuilderBeforeExplicitlyResolvingAllChildren()
    {
        $builder = new FormBuilder('name', null, new EventDispatcher(), (new FormFactoryBuilder())->getFormFactory());
        $builder->add('submit', SubmitType::class);

        $this->assertInstanceOf(ButtonBuilder::class, $builder->get('submit'));
    }
}
