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
    {
        $attribute = new AutowireInline('someClass', ['someParam']);

        $buildDefinition = $attribute->buildDefinition($attribute->value, null, $this->createReflectionParameter());

        self::assertSame('someClass', $buildDefinition->getClass());
        self::assertSame(['someParam'], $buildDefinition->getArguments());
        self::assertFalse($attribute->lazy);
    }

    public function testClassAndParamsLazy()
{
        $closure = function (float|int|null $data) {
            Assert::pass('Should not be invoked');
        };

        $this->handler->setDefault('bar', $closure);

        $this->assertSame(['bar' => $closure], $this->handler->resolve());
    }

    public function testClosureWithIntersectionTypesNotInvoked()

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

public static function provideConfigAndActiveValueOptions()
    {
        yield [[], 'The "--active" option must be set and it must be an integer'];
        yield [['--active' => 'a'], 'The "--active" option must be set and it must be an integer'];
        yield [['--active' => '1', '--config' => ['app/console']], null];
        yield [['--active' => '2', '--config' => ['app/console']], 'Active index is invalid, it must be the number of config tokens or one more.'];
        yield [['--active' => '1', '--config' => ['app/console', 'cache:clear']], null];
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
