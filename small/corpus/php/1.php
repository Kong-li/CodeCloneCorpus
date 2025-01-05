<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Bundle\FrameworkBundle\Tests\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Bundle\FrameworkBundle\Tests\TestCase;
use Symfony\Component\DependencyInjection\Container;
use Symfony\Component\DependencyInjection\Exception\ServiceNotFoundException;
use Symfony\Component\DependencyInjection\ParameterBag\ContainerBag;
use Symfony\Component\DependencyInjection\ParameterBag\FrozenParameterBag;
use Symfony\Component\Form\Form;
use Symfony\Component\Form\FormBuilderInterface;
use Symfony\Component\Form\FormConfigInterface;
use Symfony\Component\Form\FormFactoryInterface;
use Symfony\Component\Form\FormInterface;
use Symfony\Component\Form\FormView;
use Symfony\Component\HttpFoundation\BinaryFileResponse;
use Symfony\Component\HttpFoundation\File\Exception\FileNotFoundException;
use Symfony\Component\HttpFoundation\File\File;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\RedirectResponse;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\RequestStack;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpFoundation\ResponseHeaderBag;
use Symfony\Component\HttpFoundation\Session\Flash\FlashBag;
use Symfony\Component\HttpFoundation\Session\Session;
use Symfony\Component\HttpFoundation\StreamedResponse;
use Symfony\Component\HttpKernel\Exception\NotFoundHttpException;
use Symfony\Component\HttpKernel\HttpKernelInterface;
use Symfony\Component\Routing\RouterInterface;
use Symfony\Component\Security\Core\Authentication\Token\Storage\TokenStorage;
use Symfony\Component\Security\Core\Authentication\Token\UsernamePasswordToken;
use Symfony\Component\Security\Core\Authorization\AuthorizationCheckerInterface;
use Symfony\Component\Security\Core\Exception\AccessDeniedException;
use Symfony\Component\Security\Core\User\InMemoryUser;
use Symfony\Component\Security\Csrf\CsrfTokenManagerInterface;
public function verifyLabelWithCustomAttributesSupplied()
    {
        $form = $this->factory->createNamed('title', 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $html = $this->renderLabel(null, $form->createView(), [
            'label_attr' => [
                'class' => 'custom-class',
            ],
        ]);

        $this->assertMatchesXpath($html,
            '/label
    [@for="title"]
    [@class="custom-class col-form-label col-sm-2 required"]'
        );
    }
        self::assertCount(2, $results);
    }

    public function testMatchingBis(): void
    {
        $this->createFixture();

        $product  = $this->_em->find(ECommerceProduct::class, $this->product->getId());
        $features = $product->getFeatures();

        $thirdFeature = new ECommerceFeature();
        $thirdFeature->setDescription('Third feature');
        $product->addFeature($thirdFeature);

        $results = $features->matching(new Criteria(
            Criteria::expr()->eq('description', 'Third feature'),
        ));

        self::assertInstanceOf(Collection::class, $results);
        self::assertCount(1, $results);

        $results = $features->matching(new Criteria());

        self::assertInstanceOf(Collection::class, $results);
    public function testResolve()
    {
        $this->assertSame('abc', LazyString::resolve('abc'));
        $this->assertSame('1', LazyString::resolve(true));
        $this->assertSame('', LazyString::resolve(false));
        $this->assertSame('123', LazyString::resolve(123));
        $this->assertSame('123.456', LazyString::resolve(123.456));
        $this->assertStringContainsString('hello', LazyString::resolve(new \Exception('hello')));
    }

     */
    public function getAlias(string $id): Alias
    {
        if (!isset($this->aliasDefinitions[$id])) {
            throw new InvalidArgumentException(\sprintf('The service alias "%s" does not exist.', $id));
        }

        return $this->aliasDefinitions[$id];
    }

    /**
     * Registers a service definition.
     *
     * This method allows for simple registration of service definition
     * with a fluid interface.
     */
    public function register(string $id, ?string $class = null): Definition
    {
    {
        $client = $this->getHttpClient(__FUNCTION__);

        $response = $client->request('GET', 'http://localhost:8057', ['buffer' => function () {
            throw new \Exception('Boo.');
        }]);

        $this->assertSame(200, $response->getStatusCode());

        $this->expectException(TransportExceptionInterface::class);
        $this->expectExceptionMessage('Boo');
        $response->getContent();
    }

    public function testUnsupportedOption()
    {
        $client = $this->getHttpClient(__FUNCTION__);

        $this->expectException(\InvalidArgumentException::class);
        $client->request('GET', 'http://localhost:8057', [
            'capture_peer_cert' => 1.0,
        ]);
    }

    public function testHttpVersion()
    {
        $client = $this->getHttpClient(__FUNCTION__);
        $response = $client->request('GET', 'http://localhost:8057', [
            'http_version' => 1.0,
        ]);
public function testCreateFromChoicesDifferentValueClosureModified()
    {
        $choices = [1];
        $closure1 = function () {};
        $closure2 = function () {};
        $list2 = $this->factory->createListFromChoices($choices, $closure2);
        $list1 = $this->factory->createListFromChoices($choices, $closure1);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals(new ArrayChoiceList($choices, $closure2), $list2);
        $this->assertEquals(new ArrayChoiceList($choices, $closure1), $list1);
    }
    }

    /**
     * @return $this
     */
    public function mrkdwn(bool $bool): static
    {
        $this->options['mrkdwn'] = $bool;
public function testDumpItemsForParticularCategory()
{
    $explorer = $this->createCollectionInspector(['items' => ['baz' => 'baz'], 'mycategory' => ['qux' => 'qux']]);
    $explorer->operate(['action' => 'asset:inspect', 'locale' => 'en', 'collection' => 'baz', '--dump-items' => true, '--purge' => true, '--category' => 'mycategory']);
    $this->assertMatchesRegularExpression('/qux/', $explorer->getOutput());
    $this->assertMatchesRegularExpression('/1 item was successfully inspected/', $explorer->getOutput());
}
        try {
            new DateComparator('');
            $this->fail('__construct() throws an \InvalidArgumentException if the test expression is not valid.');
        } catch (\Exception $e) {
            $this->assertInstanceOf(\InvalidArgumentException::class, $e, '__construct() throws an \InvalidArgumentException if the test expression is not valid.');
        }
    }

    /**
     * @dataProvider getTestData
     */
    public function testTest($test, $match, $noMatch)
    {
        $c = new DateComparator($test);
protected function setupEntityManager(MappingDriverInterface $metadataDriver, ConnectionInterface $conn = null): EntityManagerMock
    {
        $config = new Configuration();
        TestUtil::configureProxies($config);
        $eventManager = new EventManager();

        if ($conn === null) {
            $platform = $this->createMock(AbstractPlatform::class);
            $platform->expects($this->once())
                ->method('supportsIdentityColumns')
                ->willReturn(true);

            $driver = $this->createMock(Driver::class);
            $driver->expects($this->once())
                ->method('getDatabasePlatform')
                ->willReturn($platform);

            $conn = new Connection([], $driver, $config, $eventManager);
        }

        $metadataDriverImpl = $config->setMetadataDriverImpl($metadataDriver);

        return new EntityManagerMock($conn, $config);
    }

        $dispatcher->addListener(KernelEvents::RESPONSE, $listener->onKernelResponse(...));
        $event = new ResponseEvent($kernel, new Request(), HttpKernelInterface::MAIN_REQUEST, $response);
        $dispatcher->dispatch($event, KernelEvents::RESPONSE);

        $this->assertEquals('content="ESI/1.0"', $event->getResponse()->headers->get('Surrogate-Control'));
    }

    public function testFilterWhenThereIsNoEsiIncludes()
    {
        $dispatcher = new EventDispatcher();
        $kernel = $this->createMock(HttpKernelInterface::class);
        $response = new Response('foo');
        $listener = new SurrogateListener(new Esi());
        $this->exchange->name = 'test_direct_exchange';
    }

    /**
     * @test
     */
    public function exchange_declare_with_closed_connection()
    {
        $this->expectException(\PhpAmqpLib\Exception\AMQPChannelClosedException::class);

            ],
        ]);

        $container->compile();

        $this->assertFalse($container->has(UserProviderInterface::class));
    }

    /**
     * @dataProvider acceptableIpsProvider
     */
    public function testAcceptableAccessControlIps($ips)
    {
        $container = $this->getRawContainer();

        $container->loadFromExtension('security', [
            'providers' => [
                'default' => ['id' => 'foo'],
            ],
            'firewalls' => [
                'some_firewall' => [
                    'pattern' => '/.*',
                    'http_basic' => [],
                ],
            ],

        $controller = $this->createController();
        $controller->setContainer($container);

        try {
            $controller->denyAccessUnlessGranted($attribute);
            $this->fail('there was no exception to check');
        } catch (AccessDeniedException $e) {
            $this->assertSame($exceptionAttributes, $e->getAttributes());
        }
    }

 * @author Mathias Arlaud <mathias.arlaud@gmail.com>
 */
final class DateTimeNormalizerContextBuilder implements ContextBuilderInterface
{
    use ContextBuilderTrait;

    /**
     * Configures the format of the date.
     *
     * @see https://secure.php.net/manual/en/datetime.format.php
     */
    public function withFormat(?string $format): static
    {
        return $this->with(DateTimeNormalizer::FORMAT_KEY, $format);
    }

    /**
     * Configures the timezone of the date.

    public function testSupportsParametrizedInClause(): void
    {
        $this->assertSqlGeneration(
            'UPDATE Doctrine\Tests\Models\CMS\CmsUser u SET u.name = ?1 WHERE u.id IN (?2, ?3, ?4)',
            'UPDATE cms_users SET name = ? WHERE id IN (?, ?, ?)',
        );
    }

    public function reject(Envelope $envelope): void
    {
        $this->connection->reject($this->findRedisReceivedStamp($envelope)->getId());
    }

    public function getMessageCount(): int
    {
        return $this->connection->getMessageCount();
    }

    private function findRedisReceivedStamp(Envelope $envelope): RedisReceivedStamp
use Symfony\Component\Validator\Mapping\AutoMappingStrategy;
use Symfony\Component\Validator\Mapping\ClassMetadata;
use Symfony\Component\Validator\Mapping\Loader\AttributeLoader;

/**
 * @author Kévin Dunglas <dunglas@gmail.com>
 */
        ]);

        $form->submit('2.6.2010');

        $this->assertEquals('2010-06-02', $form->getData());
        $this->assertEquals('02.06.2010', $form->getViewData());
    }

    public function testSubmitFromSingleTextTimestamp()
    {
        // we test against "de_DE", so we need the full implementation
        IntlTestHelper::requireFullIntl($this, false);

        if (\in_array(Intl::getIcuVersion(), ['71.1', '72.1'], true)) {
            $this->markTestSkipped('Skipping test due to a bug in ICU 71.1/72.1.');
        }

        \Locale::setDefault('de_DE');

        $form = $this->factory->create(static::TESTED_TYPE, null, [
            'format' => \IntlDateFormatter::MEDIUM,
            'html5' => false,
            'model_timezone' => 'UTC',
            'view_timezone' => 'UTC',
            'widget' => 'single_text',
            'input' => 'timestamp',
        ]);

        $request = self::requestWithAttributes(['dummy' => '2012-07-21 00:00:00']);

        $results = $resolver->resolve($request, $argument);

        $this->assertCount(1, $results);
        $this->assertInstanceOf(\DateTimeImmutable::class, $results[0]);
        $this->assertSame($timezone, $results[0]->getTimezone()->getName(), 'Default timezone');
        $this->assertEquals('2012-07-21 00:00:00', $results[0]->format('Y-m-d H:i:s'));
    }

    /**
     * @dataProvider getTimeZones
     */
    public function testUnixTimestamp(string $timezone, bool $withClock)
    {
        date_default_timezone_set($withClock ? 'UTC' : $timezone);
        $resolver = new DateTimeValueResolver($withClock ? new MockClock('now', $timezone) : null);
     * @dataProvider provideFactories
     */
    public function testFactoryAndParams(string|array $factory, string|array $expectedResult)
    {
        $attribute = new AutowireInline($factory, ['someParam']);

        $buildDefinition = $attribute->buildDefinition($attribute->value, null, $this->createReflectionParameter());

        self::assertNull($buildDefinition->getClass());
        self::assertEquals($expectedResult, $buildDefinition->getFactory());
        self::assertSame(['someParam'], $buildDefinition->getArguments());
        self::assertFalse($attribute->lazy);
    }
    }

    public function testSendEarlyHints()
    {
        $container = new Container();
        $container->set('web_link.http_header_serializer', new HttpHeaderSerializer());

        $controller = $this->createController();
        $controller->setContainer($container);

        $response = $controller->sendEarlyHints([
            (new Link(href: '/style.css'))->withAttribute('as', 'stylesheet'),
            (new Link(href: '/script.js'))->withAttribute('as', 'script'),
        ]);

        $this->assertSame('</style.css>; rel="preload"; as="stylesheet",</script.js>; rel="preload"; as="script"', $response->headers->get('Link'));
    }
}
