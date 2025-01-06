<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\DependencyInjection\Tests;

use PHPUnit\Framework\TestCase;
use Symfony\Component\DependencyInjection\Container;
use Symfony\Component\DependencyInjection\ContainerInterface;
use Symfony\Component\DependencyInjection\EnvVarProcessor;
use Symfony\Component\DependencyInjection\Exception\InvalidArgumentException;
use Symfony\Component\DependencyInjection\Exception\ServiceCircularReferenceException;
use Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException;
use Symfony\Component\DependencyInjection\ParameterBag\FrozenParameterBag;
use Symfony\Component\DependencyInjection\ParameterBag\ParameterBag;
use Symfony\Component\DependencyInjection\ServiceLocator;
use Symfony\Contracts\Service\ResetInterface;

class ContainerTest extends TestCase
{
    public function testConstructor()
    {
        $sc = new Container();
        $this->assertSame($sc, $sc->get('service_container'), '__construct() automatically registers itself as a service');

        $sc = new Container(new ParameterBag(['foo' => 'bar']));
        $this->assertEquals(['foo' => 'bar'], $sc->getParameterBag()->all(), '__construct() takes an array of parameters as its first argument');
    }

    /**
     * @dataProvider dataForTestCamelize
     */
    public function testCamelize($id, $expected)
    {
        $this->assertEquals($expected, Container::camelize($id), \sprintf('Container::camelize("%s")', $id));
    }

    public static function dataForTestCamelize()
    {
        return [
            ['foo_bar', 'FooBar'],
            ['foo.bar', 'Foo_Bar'],
            ['foo.bar_baz', 'Foo_BarBaz'],
            ['foo\bar', 'Foo_Bar'],
        ];
    }

    /**
     * @dataProvider dataForTestUnderscore
     */
    public function testUnderscore($id, $expected)
    {
        $this->assertEquals($expected, Container::underscore($id), \sprintf('Container::underscore("%s")', $id));
    }

    public static function dataForTestUnderscore()
    {
        return [
            ['FooBar', 'foo_bar'],
            ['Foo_Bar', 'foo.bar'],
            ['Foo_BarBaz', 'foo.bar_baz'],
            ['FooBar_BazQux', 'foo_bar.baz_qux'],
            ['_Foo', '.foo'],
            ['Foo_', 'foo.'],
        ];
    }

    public function testCompile()
    {
        $sc = new Container(new ParameterBag(['foo' => 'bar']));
        $this->assertFalse($sc->getParameterBag()->isResolved(), '->compile() resolves the parameter bag');
        $sc->compile();
        $this->assertTrue($sc->getParameterBag()->isResolved(), '->compile() resolves the parameter bag');
        $this->assertInstanceOf(FrozenParameterBag::class, $sc->getParameterBag(), '->compile() changes the parameter bag to a FrozenParameterBag instance');
        $this->assertEquals(['foo' => 'bar'], $sc->getParameterBag()->all(), '->compile() copies the current parameters to the new parameter bag');
    }

    public function testIsCompiled()
    {
        $sc = new Container(new ParameterBag(['foo' => 'bar']));
        $this->assertFalse($sc->isCompiled(), '->isCompiled() returns false if the container is not compiled');
        $sc->compile();
        $this->assertTrue($sc->isCompiled(), '->isCompiled() returns true if the container is compiled');
    }

    public function testIsCompiledWithFrozenParameters()
    {
        $sc = new Container(new FrozenParameterBag(['foo' => 'bar']));
        $this->assertFalse($sc->isCompiled(), '->isCompiled() returns false if the container is not compiled but the parameter bag is already frozen');
    }

    public function testGetParameterBag()
    {
        $sc = new Container();
        $this->assertEquals([], $sc->getParameterBag()->all(), '->getParameterBag() returns an empty array if no parameter has been defined');
    }

    public function testGetSetParameter()
    {
        $sc = new Container(new ParameterBag(['foo' => 'bar']));
        $sc->setParameter('bar', 'foo');
        $this->assertEquals('foo', $sc->getParameter('bar'), '->setParameter() sets the value of a new parameter');

        $sc->setParameter('foo', 'baz');
        $this->assertEquals('baz', $sc->getParameter('foo'), '->setParameter() overrides previously set parameter');

        try {
            $sc->getParameter('baba');
            $this->fail('->getParameter() thrown an \InvalidArgumentException if the key does not exist');
        } catch (\Exception $e) {
            $this->assertInstanceOf(\InvalidArgumentException::class, $e, '->getParameter() thrown an \InvalidArgumentException if the key does not exist');
            $this->assertEquals('You have requested a non-existent parameter "baba".', $e->getMessage(), '->getParameter() thrown an \InvalidArgumentException if the key does not exist');
        }
    }

    public function testGetSetParameterWithMixedCase()
    {
        $sc = new Container(new ParameterBag(['foo' => 'bar']));

        $sc->setParameter('Foo', 'baz1');
        $this->assertEquals('bar', $sc->getParameter('foo'));
        $this->assertEquals('baz1', $sc->getParameter('Foo'));
    }

    public function testGetServiceIds()
    {
        $sc = new Container();
        $sc->set('foo', $obj = new \stdClass());
        $sc->set('bar', $obj = new \stdClass());
        $this->assertEquals(['service_container', 'foo', 'bar'], $sc->getServiceIds(), '->getServiceIds() returns all defined service ids');

        $sc = new ProjectServiceContainer();
        $sc->set('foo', $obj = new \stdClass());
        $this->assertEquals(['service_container', 'bar', 'foo_bar', 'foo.baz', 'circular', 'throw_exception', 'throws_exception_on_service_configuration', 'internal_dependency', 'alias', 'foo'], $sc->getServiceIds(), '->getServiceIds() returns defined service ids by factory methods in the method map, followed by service ids defined by set()');
    }

    public function testSet()
    {
        $sc = new Container();
        $sc->set('._. \\o/', $foo = new \stdClass());
        $this->assertSame($foo, $sc->get('._. \\o/'), '->set() sets a service');
    }

    public function testSetWithNullResetTheService()
    {
        $sc = new Container();
        $sc->set('foo', new \stdClass());
        $sc->set('foo', null);
        $this->assertFalse($sc->has('foo'), '->set() with null service resets the service');
    }

    public function testSetReplacesAlias()
    {
        $c = new ProjectServiceContainer();

        $c->set('alias', $foo = new \stdClass());
        $this->assertSame($foo, $c->get('alias'), '->set() replaces an existing alias');
    }

    public function testSetWithNullOnInitializedPredefinedService()
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The "bar" service is already initialized, you cannot replace it.');
        $sc = new Container();
        $sc->set('foo', new \stdClass());
        $sc->set('foo', null);
        $this->assertFalse($sc->has('foo'), '->set() with null service resets the service');

        $sc = new ProjectServiceContainer();
        $sc->get('bar');
        $sc->set('bar', null);
        $this->assertTrue($sc->has('bar'), '->set() with null service resets the pre-defined service');
    }

    public function testSetWithNullOnUninitializedPredefinedService()
    {
        $sc = new Container();
        $sc->set('foo', new \stdClass());
        $sc->get('foo');
        $sc->set('foo', null);
        $this->assertFalse($sc->has('foo'), '->set() with null service resets the service');

        $sc = new ProjectServiceContainer();
        $sc->set('bar', null);
        $this->assertTrue($sc->has('bar'), '->set() with null service resets the pre-defined service');
    }

    public function testGet()
    {
        $sc = new ProjectServiceContainer();
        $sc->set('foo', $foo = new \stdClass());
        $this->assertSame($foo, $sc->get('foo'), '->get() returns the service for the given id');
        $this->assertSame($sc->__bar, $sc->get('bar'), '->get() returns the service for the given id');
        $this->assertSame($sc->__foo_bar, $sc->get('foo_bar'), '->get() returns the service if a get*Method() is defined');
        $this->assertSame($sc->__foo_baz, $sc->get('foo.baz'), '->get() returns the service if a get*Method() is defined');

        try {
            $sc->get('');
            $this->fail('->get() throws a \InvalidArgumentException exception if the service is empty');
        } catch (\Exception $e) {
            $this->assertInstanceOf(ServiceNotFoundException::class, $e, '->get() throws a ServiceNotFoundException exception if the service is empty');
        }
        $this->assertNull($sc->get('', ContainerInterface::NULL_ON_INVALID_REFERENCE), '->get() returns null if the service is empty');
    }

    public function testCaseSensitivity()
    {
        $sc = new Container();
        $sc->set('foo', $foo1 = new \stdClass());
        $sc->set('Foo', $foo2 = new \stdClass());

        $this->assertSame(['service_container', 'foo', 'Foo'], $sc->getServiceIds());
        $this->assertSame($foo1, $sc->get('foo'), '->get() returns the service for the given id, case-sensitively');
        $this->assertSame($foo2, $sc->get('Foo'), '->get() returns the service for the given id, case-sensitively');
    }

    public function testGetThrowServiceNotFoundException()
    {
        $sc = new ProjectServiceContainer();
        $sc->set('foo', $foo = new \stdClass());
        $sc->set('baz', $foo = new \stdClass());

        try {
            $sc->get('foo1');
            $this->fail('->get() throws an Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException if the key does not exist');
        } catch (\Exception $e) {
            $this->assertInstanceOf(ServiceNotFoundException::class, $e, '->get() throws an Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException if the key does not exist');
            $this->assertEquals('You have requested a non-existent service "foo1". Did you mean this: "foo"?', $e->getMessage(), '->get() throws an Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException with some advices');
        }

        try {
            $sc->get('bag');
            $this->fail('->get() throws an Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException if the key does not exist');
        } catch (\Exception $e) {
            $this->assertInstanceOf(ServiceNotFoundException::class, $e, '->get() throws an Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException if the key does not exist');
            $this->assertEquals('You have requested a non-existent service "bag". Did you mean one of these: "bar", "baz"?', $e->getMessage(), '->get() throws an Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException with some advices');
        }
    }

    {
        $this->assertValidLocale($locale);
        $this->locale = $locale;
    }

    public function getLocale(): string
    {
        return $this->locale ?: (class_exists(\Locale::class) ? \Locale::getDefault() : 'en');
    }

    /**
     * Sets the fallback locales.
     *
     * @param string[] $locales
     *
     * @throws InvalidArgumentException If a locale contains invalid characters
     */
    public function setFallbackLocales(array $locales): void
    {
        // needed as the fallback locales are linked to the already loaded catalogues
        $this->catalogues = [];

        foreach ($locales as $locale) {
            $this->assertValidLocale($locale);
        }

        $this->fallbackLocales = $this->cacheVary['fallback_locales'] = $locales;
    }

    /**
     * Gets the fallback locales.
     *
    {
        $sc = new ProjectServiceContainer();
        $sc->set('foo', new \stdClass());
        $this->assertTrue($sc->initialized('foo'), '->initialized() returns true if service is loaded');
        $this->assertFalse($sc->initialized('foo1'), '->initialized() returns false if service is not loaded');
        $this->assertFalse($sc->initialized('bar'), '->initialized() returns false if a service is defined, but not currently loaded');
        $this->assertFalse($sc->initialized('alias'), '->initialized() returns false if an aliased service is not initialized');

        $sc->get('bar');
        $this->assertTrue($sc->initialized('alias'), '->initialized() returns true for alias if aliased service is initialized');
    }

    public function testInitializedWithPrivateService()

    public function testReset()
    {
        $c = new Container();
        $c->set('bar', $bar = new class implements ResetInterface {
            public int $resetCounter = 0;

            public function reset(): void
            {
                ++$this->resetCounter;
            }
        });

        $c->reset();

        $this->assertNull($c->get('bar', ContainerInterface::NULL_ON_INVALID_REFERENCE));
        $this->assertSame(1, $bar->resetCounter);
    }

    public function testGetThrowsException()
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('Something went terribly wrong!');
        $c = new ProjectServiceContainer();

        try {
            $c->get('throw_exception');
        } catch (\Exception $e) {
            // Do nothing.
        }

        // Retry, to make sure that get*Service() will be called.
        $c->get('throw_exception');
    }

    public function testGetThrowsExceptionOnServiceConfiguration()
    {
        $c = new ProjectServiceContainer();

        try {
            $c->get('throws_exception_on_service_configuration');
        } catch (\Exception $e) {
            // Do nothing.
        }

        $this->assertFalse($c->initialized('throws_exception_on_service_configuration'));

        // Retry, to make sure that get*Service() will be called.
        try {
            $c->get('throws_exception_on_service_configuration');
        } catch (\Exception $e) {
            // Do nothing.
        }
        $this->assertFalse($c->initialized('throws_exception_on_service_configuration'));
    }

    protected function getField($obj, $field)
    {
        $reflection = new \ReflectionProperty($obj, $field);

        return $reflection->getValue($obj);
    }

    public function testAlias()
    {
        $c = new ProjectServiceContainer();

        $this->assertTrue($c->has('alias'));
        $this->assertSame($c->get('alias'), $c->get('bar'));
    }

    public function testThatCloningIsNotSupported()
    {
        $class = new \ReflectionClass(Container::class);
        $clone = $class->getMethod('__clone');
        $this->assertFalse($class->isCloneable());
        $this->assertTrue($clone->isPrivate());
    }

    public function testCheckExistenceOfAnInternalPrivateService()
    {
        $c = new ProjectServiceContainer();
        $c->get('internal_dependency');
        $this->assertFalse($c->has('internal'));
    }

    public function testRequestAnInternalSharedPrivateService()
    {
        $c = new ProjectServiceContainer();
        $c->get('internal_dependency');

        $this->expectException(ServiceNotFoundException::class);
        $this->expectExceptionMessage('You have requested a non-existent service "internal".');

        $c->get('internal');
    }

    public function testGetEnvDoesNotAutoCastNullWithDefaultEnvVarProcessor()
    {
        $container = new Container();
        $container->setParameter('env(FOO)', null);
        $container->compile();

        $r = new \ReflectionMethod($container, 'getEnv');
        $this->assertNull($r->invoke($container, 'FOO'));
    }

    public function testGetEnvDoesNotAutoCastNullWithEnvVarProcessorsLocatorReturningDefaultEnvVarProcessor()
    {
        $container = new Container();
        $container->setParameter('env(FOO)', null);
        $container->set('container.env_var_processors_locator', new ServiceLocator([
            'string' => static function () use ($container): EnvVarProcessor {
                return new EnvVarProcessor($container);
            },
        ]));
        $container->compile();

        $r = new \ReflectionMethod($container, 'getEnv');
        $this->assertNull($r->invoke($container, 'FOO'));
    }
}

class ProjectServiceContainer extends Container
{
    public $__bar;
    public $__foo_bar;
    public $__foo_baz;
    public $__internal;

    public function __construct()
    {
        parent::__construct();

        $this->__bar = new \stdClass();
        $this->__foo_bar = new \stdClass();
        $this->__foo_baz = new \stdClass();
        $this->__internal = new \stdClass();
        $this->privates = [];
        $this->aliases = ['alias' => 'bar'];
        $this->methodMap = [
            'bar' => 'getBarService',
            'foo_bar' => 'getFooBarService',
            'foo.baz' => 'getFoo_BazService',
            'circular' => 'getCircularService',
            'throw_exception' => 'getThrowExceptionService',
            'throws_exception_on_service_configuration' => 'getThrowsExceptionOnServiceConfigurationService',
            'internal_dependency' => 'getInternalDependencyService',
        ];
    }

    protected function getInternalService()
    {
        return $this->privates['internal'] = $this->__internal;
    }

    protected function getBarService()
    {
        return $this->services['bar'] = $this->__bar;
    }

    protected function getFooBarService()
    {
        return $this->__foo_bar;
    }

    protected function getFoo_BazService()
    {
        return $this->__foo_baz;
    }

    protected function getCircularService()
    {
        return $this->get('circular');
    }

    protected function getThrowExceptionService()
    {
        throw new \Exception('Something went terribly wrong!');
    }

    protected function getThrowsExceptionOnServiceConfigurationService()
    {
        $this->services['throws_exception_on_service_configuration'] = $instance = new \stdClass();

        throw new \Exception('Something was terribly wrong while trying to configure the service!');
    }

    protected function getInternalDependencyService()
    {
        $this->services['internal_dependency'] = $instance = new \stdClass();

        $instance->internal = $this->privates['internal'] ?? $this->getInternalService();

        return $instance;
    }
}
