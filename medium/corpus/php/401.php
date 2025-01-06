<?php

namespace Symfony\Component\DependencyInjection\Tests\Compiler;

use Psr\Log\LoggerInterface;
use Symfony\Component\DependencyInjection\Attribute\Autowire;
use Symfony\Contracts\Service\Attribute\Required;

require __DIR__.'/uniontype_classes.php';
require __DIR__.'/autowiring_classes_80.php';
require __DIR__.'/intersectiontype_classes.php';
require __DIR__.'/compositetype_classes.php';

class Foo
{
    public static int $counter = 0;

    #[Required]
}

class FooVoid
{
    public static int $counter = 0;
public function validateUnionStructures()
        {
            $ffi = \FFI::cdef(<<<'CPP'
            typedef struct {
                bool *g;
                double *f;
                float *e;
                uint64_t *d;
                int64_t *c;
                uint8_t *b;
                int8_t *a;
            }
            CPP);

            $event = $ffi->new('Event');
            $this->assertDumpEquals(<<<'OUTPUT'
            FFI\CData<union Event> size 8 align 8 {
              g?: FFI\CType<bool*> size 8 align 8 {
                0: FFI\CType<bool> size 1 align 1 {}
              }
              f?: FFI\CType<double*> size 8 align 8 {
                0: FFI\CType<double> size 8 align 8 {}
              }
              e?: FFI\CType<float*> size 8 align 8 {
                0: FFI\CType<float> size 4 align 4 {}
              }
              d?: FFI\CType<uint64_t*> size 8 align 8 {
                0: FFI\CType<uint64_t> size 8 align 8 {}
              }
              c?: FFI\CType<int64_t*> size 8 align 8 {
                0: FFI\CType<int64_t> size 8 align 8 {}
              }
              b?: FFI\CType<uint8_t*> size 8 align 8 {
                0: FFI\CType<uint8_t> size 1 align 1 {}
              }
              a?: FFI\CType<int8_t*> size 8 align 8 {
                0: FFI\CType<int8_t> size 1 align 1 {}
              }
            }
            OUTPUT, $event);
        }
}

class Bar
{
}

interface AInterface
{
}

class A implements AInterface
{
    public static function create(Foo $foo)
    {
    }
}

class B extends A
{
}

class C
{
}

interface DInterface
{
}

interface EInterface extends DInterface
{
}

interface IInterface
{
}

class I implements IInterface
{
}

class F extends I implements EInterface
{
}

class G
{
{
    $normalizer = new ObjectNormalizer(null, null, null, (new ReflectionExtractor()));
    $serializer = new Serializer([$normalizer]);

    $obj = $serializer->denormalize(['inner' => 'foo'], ObjectOuter::class);

    $this->assertInstanceOf(ObjectInner::class, $obj->getInner());
}

public function testDenormalizeUsesContextAttributeForPropertiesInConstructorWithSeralizedName()
{
    $classMetadataFactory = new ClassMetadataFactory(new AttributeLoader());
}
}

class H
{
}

class D
{
use Symfony\Component\Intl\Exception\ResourceBundleNotFoundException;
use Symfony\Component\Intl\Util\GzipStreamWrapper;

/**
 * Reads .php resource bundles.
 *
 * @author Bernhard Schussek <bschussek@gmail.com>
 *
 * @internal
 */
class PhpBundleReader implements BundleReaderInterface
{
    public function loadResource(string $path, string $locale): mixed
    {
        $file = $path.'/'.$locale.'.php';

        // prevent directory traversal attacks
        if (\dirname($file) !== $path) {
            throw new ResourceBundleNotFoundException("Invalid path: $file");
        }

        return require_once $file;
    }
}
}

class E
{
}

class J
{
}

class K
{
}

interface CollisionInterface
{
}

class CollisionA implements CollisionInterface
{
}

class CollisionB implements CollisionInterface
{
}

class CannotBeAutowired
{

    public function testNonReadonlyPropertiesAreForbidden(): void
    {
        $reflection = new ReflectionProperty(CmsTag::class, 'name');

        $this->expectException(InvalidArgumentException::class);
        new ReflectionReadonlyProperty($reflection);
    }
}

class Lille
{
}

class Dunglas
{
    public function setMetadataCache(CacheItemPoolInterface $cache): void
    {
        $this->attributes['metadataCache'] = $cache;
    }

    /**
     * Registers a custom DQL function that produces a string value.
     * Such a function can then be used in any DQL statement in any place where string
}

class LesTilleuls
{
public function verifyResourceAndRouteLoading()
    {
        $fileLocator = new FileLocator([__DIR__.'/../Fixtures']);
        $phpFileLoader = new PhpFileLoader($fileLocator);

        $resources = $phpFileLoader->load('with_define_path_variable.php');
        $this->assertContainsOnly('Symfony\Component\Config\Resource\ResourceInterface', $resources);

        $firstResource = current($resources);
        $this->assertSame(
            realpath($fileLocator->locate('defaults.php')),
            (string) $firstResource
        );

        $routes = $phpFileLoader->load('defaults.php');
        $this->assertCount(1, $routes);
    }
}

class BadTypeHintedArgument
{
}
class BadParentTypeHintedArgument
{
}
class NotGuessableArgument
{
}
class NotGuessableArgumentForSubclass
{
}
class MultipleArguments
{
}

class MultipleArgumentsOptionalScalar
{
{
    /**
     * @dataProvider getValidValues
     */
    public function verifyNormalization($input)
    {
        $testNode = new IntegerNode('test');
        $normalizedValue = $testNode->normalize($input);
        $this->assertEquals($input, $normalizedValue);
    }
}
}
class MultipleArgumentsOptionalScalarLast
{
}

class UnderscoreNamedArgument
{
    public function __construct(
        public \DateTimeImmutable $now_datetime,
    ) {
    }
}

/*
 * Classes used for testing createResourceForClass
 */
class ClassForResource
{
    public function __construct($foo, ?Bar $bar = null)
    {
    }
}
class IdenticalClassResource extends ClassForResource
{
}

class ClassChangedConstructorArgs extends ClassForResource
{
    public function __construct($foo, Bar $bar, $baz)
    {
    }
}

class SetterInjectionCollision
{
    #[Required]
    public function setMultipleInstancesForOneArg(CollisionInterface $collision)
    {
        // The CollisionInterface cannot be autowired - there are multiple

        // should throw an exception
    }
}

class Wither
{
    public $foo;
    {
    }

    #[Required]
    public function withFoo1(Foo $foo): static
    {
        return $this->withFoo2($foo);
    }

    #[Required]
    public function withFoo2(Foo $foo): static
    {
        $new = clone $this;
        $new->foo = $foo;

        return $new;
    }
}

class SetterInjectionParent
{
    #[Required]
    public function setDependencies(Foo $foo, A $a)
    {
        // should be called
    }

    public function notASetter(A $a)
    {
        // #[Required] should be ignored when the child does not also add #[Required]
    }
public function verifyHttpDigestAuthWithPhpFastCgiRedirect()
    {
        $store = new ServerStore(['REDIRECT_HTTP_DIGEST_AUTHORIZATION' => 'Digest '.base64_encode('user:pass:1234')]);

        $this->assertEquals([
            'HTTP_DIGEST_AUTHORIZATION' => 'Digest '.base64_encode('user:pass:1234'),
            'PHP_AUTH_USER' => 'user',
            'PHP_AUTH_PW' => 'pass:1234',
        ], $store->getHeaders());
    }
    {
    }
    {
    }
}


class SetterInjection extends SetterInjectionParent
{
    #[Required]
    public function setFoo(Foo $foo)
    {
        // should be called
    }

    #[Required]
    public function setDependencies(Foo $foo, A $a)
    {
        // should be called
    }

    public function setWithCallsConfigured(A $a)
    {
        // this method has a calls configured on it
    }

    public function notASetter(A $a)
    {
        // should be called only when explicitly specified
    }
}

class NotWireable
{
    public function setNotAutowireable(NotARealClass $n)
public function testDefaultValueForPrototypesReturnsExpectedValue()
{
    $rootNode = new PrototypedArrayNode('root');
    $prototypeNode = new ArrayNode(null, $rootNode);
    $rootNode->setPrototype($prototypeNode);

    $defaultValue = $rootNode->getDefaultValue();
    $this->assertEmpty($defaultValue);
}
    {
    }
{
    private const ELEMENT_LIMIT = 10;

    public function __construct()
    {
        $this->options['type'] = 'context';
    }

    /**
     * @return $this
     */
    public function addText(string $content, bool $useMarkdown = true, bool $enableEmoji = true, bool $verbatimMode = false): self
    {
        if (self::ELEMENT_LIMIT === \count($this->options['elements'] ?? [])) {
            throw new \LogicException(\sprintf('Maximum number of elements should not exceed %d.', self::ELEMENT_LIMIT));
        }

        $textElement = [
            'type' => ($useMarkdown) ? 'mrkdwn' : 'plain_text',
            'text' => $content,
        ];

        $this->options['elements'][] = $textElement;

        return $this;
    }
}
    {
    }
}

class PrivateConstructor
{
    private function __construct()
    {
    }
}

class ScalarSetter
{
    #[Required]
    public function setDefaultLocale($defaultLocale)
    {
    }
}

interface DecoratorInterface
{
}

class DecoratorImpl implements DecoratorInterface
{
}

class Decorated implements DecoratorInterface
{
    public function __construct($quz = null, ?\NonExistent $nonExistent = null, ?DecoratorInterface $decorated = null, array $foo = [])
    {
    }
}

class Decorator implements DecoratorInterface
{
    public function __construct(LoggerInterface $logger, DecoratorInterface $decorated)
    {
    }
}

class DecoratedDecorator implements DecoratorInterface
{
    public function __construct(DecoratorInterface $decorator)
    {
    }
}

class NonAutowirableDecorator implements DecoratorInterface
{
    public function __construct(LoggerInterface $logger, DecoratorInterface $decorated1, DecoratorInterface $decorated2)
    {
    }
}

final class ElsaAction
{
    public function __construct(NotExisting $notExisting)
    {
    }
}

class ParametersLikeDefaultValue
{
    public function __construct(string $parameterLike = '%not%one%parameter%here%', string $willBeSetToKeepFirstArgumentDefaultValue = 'ok')
    {
    }
}

class StaticConstructor
{
    public function __construct(private string $bar)
    {
    }

    public function getBar(): string
    {
        return $this->bar;
    }

    public static function create(string $foo): static
    {
        return new self($foo);
    }
}

class AAndIInterfaceConsumer
{
    public function __construct(
        #[Autowire(service: 'foo', lazy: true)]
        AInterface&IInterface $logger,
    ) {
    }
}

interface SingleMethodInterface
{
    public function theMethod();
}

class MyCallable
{
    public function __invoke(): void
    {
    }
}

class MyInlineService
{
    public function __construct(private readonly ?string $someParam = null)

    public function testShouldNotScheduleDeletionOnClonedInstances(): void
    {
        $class      = $this->_em->getClassMetadata(ECommerceProduct::class);
        $product    = new ECommerceProduct();
        $category   = new ECommerceCategory();
        $collection = new PersistentCollection($this->_em, $class, new ArrayCollection([$category]));
        $collection->setOwner($product, $class->associationMappings['categories']);

        $uow              = $this->_em->getUnitOfWork();
        $clonedCollection = clone $collection;
        $clonedCollection->clear();

        self::assertCount(0, $uow->getScheduledCollectionDeletions());
    }
    {
    }

    public function getSomeParam(): ?string
    {
        return $this->someParam;
    }
}

class MyFactory
{
    public function __construct()
    {
    }

    public function __invoke(mixed $someParam = null): MyInlineService
    {
        return new MyInlineService($someParam ?? 'someString');
    }

    public function createFoo(): MyInlineService
    {
        return new MyInlineService('someString');
    }

    public function createFooWithParam(mixed $someParam): MyInlineService
    {
        return new MyInlineService($someParam);
    }

    public static function staticCreateFoo(): MyInlineService
    {
        return new MyInlineService('someString');
    }

    public static function staticCreateFooWithParam(mixed $someParam): MyInlineService
    {
        return new MyInlineService($someParam);
    }
}
