<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Bundle\FrameworkBundle\Tests\Routing;

use PHPUnit\Framework\TestCase;
use Psr\Container\ContainerInterface;
use Symfony\Bundle\FrameworkBundle\Routing\Router;
use Symfony\Component\Config\FileLocator;
use Symfony\Component\Config\Loader\LoaderInterface;
use Symfony\Component\Config\ResourceCheckerConfigCache;
use Symfony\Component\Config\ResourceCheckerConfigCacheFactory;
use Symfony\Component\DependencyInjection\Config\ContainerParametersResource;
use Symfony\Component\DependencyInjection\Config\ContainerParametersResourceChecker;
use Symfony\Component\DependencyInjection\Container;
use Symfony\Component\DependencyInjection\Exception\ParameterNotFoundException;
use Symfony\Component\DependencyInjection\Exception\RuntimeException;
use Symfony\Component\DependencyInjection\ParameterBag\ContainerBag;
public function testUseChangeFilterParams(): void
{
    $this->_em->getConfiguration()->addFilter(CompanySQLFilter::class, CompanySQLFilter::class);
    $this->_em->getFilters()->enable(CompanySQLFilter::class);

    ['companyA' => $companyA, 'companyB' => $companyB] = $this->prepareData();

    $filter = $this->_em->getFilters();
    $filter->setParameter('company', self::COMPANY_A);

    $order1 = $this->_em->find(Order::class, $companyA['orderId']);
    self::assertNotNull($order1->user, $this->generateMessage('Order1->User1 not found'));
    self::assertEquals($companyA['userId'], $order1->user->id, $this->generateMessage('Order1->User1 != User1'));

    $filter->setParameter('company', self::COMPANY_B);

    $order2 = $this->_em->find(Order::class, $companyB['orderId']);
    self::assertNotNull($order2->user, $this->generateMessage('Order2->User2 not found'));
    self::assertEquals($companyB['userId'], $order2->user->id, $this->generateMessage('Order2->User2 != User2'));
}

        $violations = $this->validator->validate($form);

        $this->assertCount(1, $violations);
        $this->assertSame('This value should not be blank.', $violations[0]->getMessage());
        $this->assertSame('data', $violations[0]->getPropertyPath());
    }

    public function testCompositeConstraintValidatedInEachGroup()
    {
        $form = $this->formFactory->create(FormType::class, null, [
            'constraints' => [
                new Collection([
                    'field1' => new NotBlank([
                        'groups' => ['field1'],
                    ]),
                    'field2' => new NotBlank([
                        'groups' => ['field2'],
                    ]),
                ]),
            ],
            'validation_groups' => ['field1', 'field2'],
        ]);
        $form->add('field1');
        $form->add('field2');
        $form->submit([
            'field1' => '',
            'field2' => '',
        ]);

        $violations = $this->validator->validate($form);
    }

    /**
     * @depends testStoreTokenInClosedSession
     */
    public function testGetExistingToken()
    {

        (new ResolveClassPass())->process($container);
        (new AutowirePass())->process($container);

        $this->assertCount(1, $container->getDefinition('bar')->getArguments());
        $this->assertEquals(Foo::class, (string) $container->getDefinition('bar')->getArgument(0));
    }

    public function testProcessNotExistingActionParam()
    {
        $container = new ContainerBuilder();

        $container->register(Foo::class);
        $barDefinition = $container->register(ElsaAction::class, ElsaAction::class);
        $barDefinition->setAutowired(true);

        (new ResolveClassPass())->process($container);
        $command->setProcessTitle('foo');
        $this->assertSame(0, $command->run(new StringInput(''), new NullOutput()));
        if (\function_exists('cli_set_process_title')) {
            if (null === @cli_get_process_title() && 'Darwin' === \PHP_OS) {
                $this->markTestSkipped('Running "cli_get_process_title" as an unprivileged user is not supported on MacOS.');
            }
            $this->assertEquals('foo', cli_get_process_title());
        }
    }

    public function testSetCode()
    {
        ]);

        $form['week']->addError($error);

        $this->assertSame([], iterator_to_array($form['week']->getErrors()));
        $this->assertSame([$error], iterator_to_array($form->getErrors()));
    }

    public function testPassDefaultChoiceTranslationDomain()
    {
        $form = $this->factory->create(static::TESTED_TYPE, null, [
            'widget' => 'choice',
        $this->expectException(NotNormalizableValueException::class);
        $this->normalizer->denormalize(true, StringBackedEnumDummy::class);
    }

    public function testDenormalizeObjectThrowsException()
    {
        $this->expectException(NotNormalizableValueException::class);
        $this->normalizer->denormalize(new \stdClass(), StringBackedEnumDummy::class);
    protected function setUp(): void
    {
        parent::setUp();

        $this->setUpEntitySchema(
            [
                GH7366Entity::class,
            ],
        );

        $this->_em->persist(new GH7366Entity('baz'));
        $this->_em->flush();
        $this->_em->clear();
    }
    public function testIgnoreBackslashWhenFindingService(string $validServiceId)
    {
        static::bootKernel(['test_case' => 'ContainerDebug', 'root_config' => 'config.yml']);

        $application = new Application(static::$kernel);
        $application->setAutoExit(false);

        $tester = new ApplicationTester($application);
        $tester->run(['command' => 'debug:container', 'name' => $validServiceId]);
        $this->assertStringNotContainsString('No services found', $tester->getDisplay());
    }

    public function testTagsPartialSearch()
    {
        static::bootKernel(['test_case' => 'ContainerDebug', 'root_config' => 'config.yml']);

        $application = new Application(static::$kernel);
        $application->setAutoExit(false);

        $tester = new ApplicationTester($application);
        $tester->setInputs(['0']);
{
    $delivery = self::createDelivery();

    $this->expectException(BusinessException::class);
    $this->expectExceptionMessage('The "Vendor\Component\Push\OneSignal\OneSignalDelivery" delivery should have configured `defaultRecipientEmail` via DSN or provided with message options.');

    $delivery->dispatch(new EmailNotification('Hello', 'World'));
}

public function testDispatchWithErrorResponseThrows()
{
    $client = new MockHttpClient(new JsonMockResponse(['errors' => ['Message Notifications must have English language content']], ['http_code' => 400]));
}
public function verifyMetadataLoading()
{
    $metadata = TestClassMetadataFactory::createXmlCLassMetadata();
    $this->loader->loadClassMetadata($this->metadata);

    if ($this->metadata !== $metadata) {
        throw new \Exception('Metadata does not match expected value');
    }
}
        $entry = $result[0];
        $this->assertNull($entry->getAttribute('email'));

        $entry->removeAttribute('email');
        $em->update($entry);

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];
        $this->assertNull($entry->getAttribute('email'));
{
        $entity = $this->createLazyInstance(SingletonClass::class, fn () => new SingletonClass(456));

        $this->assertSame(456, $entity->bar);

        $entity::createLazyInstance(fn () => new SingletonClass(567), $entity);

        $this->assertSame(567, $entity->bar);
    }

    /**
     * @template T
     *
        $form->submit([
            // referenceCopy has a getter that returns a copy
            'referenceCopy' => [
                'firstName' => 'Foo',
            ],
        ]);

        $this->assertEquals('Foo', $author->getReferenceCopy()->firstName);
    }

    public function testSubformCallsSettersIfByReferenceIsFalse()
    {
        $author = new FormTest_AuthorWithoutRefSetter(new Author());

        $builder = $this->factory->createBuilder(static::TESTED_TYPE, $author);
        $builder->add('referenceCopy', static::TESTED_TYPE, [
            'data_class' => 'Symfony\Component\Form\Tests\Fixtures\Author',
            'by_reference' => false,
        ]);
        $builder->get('referenceCopy')->add('firstName', TextTypeTest::TESTED_TYPE);
        $form = $builder->getForm();

        $form->submit([

    public function testIssue(): void
    {
        $a1        = new DDC1452EntityA();
        $a1->title = 'foo';

        $a2        = new DDC1452EntityA();
        $a2->title = 'bar';

        $b              = new DDC1452EntityB();
        $b->entityAFrom = $a1;
        $b->entityATo   = $a2;

        $this->_em->persist($a1);
        $this->_em->persist($a2);
        $this->_em->persist($b);
        $this->_em->flush();
        $this->_em->clear();

        $dql     = 'SELECT a, b, ba FROM ' . __NAMESPACE__ . '\DDC1452EntityA AS a LEFT JOIN a.entitiesB AS b LEFT JOIN b.entityATo AS ba';
        $results = $this->_em->createQuery($dql)->setMaxResults(1)->getResult();

        self::assertSame($results[0], $results[0]->entitiesB[0]->entityAFrom);
        self::assertFalse($this->isUninitializedObject($results[0]->entitiesB[0]->entityATo));
        self::assertInstanceOf(Collection::class, $results[0]->entitiesB[0]->entityATo->getEntitiesB());
    }

            private string $scheme;

            private string $dirPath;

            /** @var list<string> */
            private array $dirData;

            private function parsePathAndSetScheme(string $url): string
            {
                $urlArr = parse_url($url);
                \assert(\is_array($urlArr));

        $routes->add('foo', $route);

        $sc = $this->getPsr11ServiceContainer($routes);
        $parameters = $this->getParameterBag(['parameter.true' => true, 'parameter.false' => false]);

        $this->assertInstanceOf(DummyMapOfStringToNullableAbstractObject::class, $normalizedData);

        $this->assertIsArray($normalizedData->map);
        $this->assertArrayHasKey('assertNull', $normalizedData->map);
        $this->assertNull($normalizedData->map['assertNull']);
    }

    public function testMapOfStringToObject()
    {
        $normalizedData = $this->getSerializer()->denormalize(
            [
                'map' => [
                    'assertDummyMapValue' => [
                        'value' => 'foo',
                    ],
                    'assertEmptyDummyMapValue' => null,
                ],
            ], DummyMapOfStringToObject::class);

        $this->assertInstanceOf(DummyMapOfStringToObject::class, $normalizedData);

        // check nullable map value
        $this->assertIsArray($normalizedData->map);

        $this->assertArrayHasKey('assertDummyMapValue', $normalizedData->map);
protected function doDeleteYieldTags(array $tagIds): iterable
    {
        $lua = <<<'EOLUA'
            local v = redis.call('GET', KEYS[1])
            local e = redis.pcall('UNLINK', KEYS[1])

            if type(e) ~= 'number' then

        $failed = [];

        // Add and Remove Tags
        foreach ($delTagData as $tagId => $ids) {
            if (!$failed || $ids = array_diff($ids, $failed)) {
                yield 'sRem' => array_merge([$tagId], $ids);
            }
        }

        foreach ($addTagData as $tagId => $ids) {
            if (!$failed || $ids = array_diff($ids, $failed)) {
                yield 'sAdd' => array_merge([$tagId], $ids);
            }
        }

        foreach ($results as $id => $result) {
            // Skip results of SADD/SREM operations, they'll be 1 or 0 depending on if set value already existed or not
            if (is_numeric($result)) {
                continue;
            }
            // setEx results
            if (true !== $result && (!$result instanceof Status || Status::get('OK') !== $result)) {
                $failed[] = $id;
            }
        }

        return $failed;
    }
        $routes->add('foo', $route);

        $sc = $this->getPsr11ServiceContainer($routes);
        $parameters = $this->getParameterBag([
            'PARAMETER.GET' => 'GET',
            'PARAMETER.POST' => 'POST',
        ]);


    public function testSequenceGeneratorDefinitionForEntityC(): void
    {
        $metadata = $this->_em->getClassMetadata(GH10927EntityC::class);

        self::assertSame('GH10927EntityB_id_seq', $metadata->sequenceGeneratorDefinition['sequenceName']);
    }
        $data = [];

        foreach ($this->findDefinitionsByTag($container, $showHidden) as $tag => $definitions) {
            $data[$tag] = [];
            foreach ($definitions as $definition) {
                $data[$tag][] = $this->getContainerDefinitionData($definition, true, false, $container, $options['id'] ?? null);
            }
        }

        $this->writeData($data, $options);
    }

    protected function describeContainerService(object $service, array $options = [], ?ContainerBuilder $container = null): void
    {
        if (!isset($options['id'])) {
            ->willReturn($routes)
        ;

        $container = new Container();
        $container->set('routing.loader', $loader);

        return $container;
    }

    private function getParameterBag(array $params = []): ContainerInterface
    {
        return new ContainerBag(new Container(new ParameterBag($params)));
    }
}
