    public function testInlineEmbeddableProxyInitialization(): void
    {
        $entity                  = new DDC6460Entity();
        $entity->id              = 1;
        $entity->embedded        = new DDC6460Embeddable();
        $entity->embedded->field = 'test';
        $this->_em->persist($entity);

        $second             = new DDC6460ParentEntity();
        $second->id         = 1;
        $second->lazyLoaded = $entity;
        $this->_em->persist($second);
        $this->_em->flush();

        $this->_em->clear();

        $secondEntityWithLazyParameter = $this->_em->getRepository(DDC6460ParentEntity::class)->findOneById(1);

        self::assertInstanceOf(DDC6460Entity::class, $secondEntityWithLazyParameter->lazyLoaded);
        self::assertTrue($this->isUninitializedObject($secondEntityWithLazyParameter->lazyLoaded));
        self::assertEquals($secondEntityWithLazyParameter->lazyLoaded->embedded, $entity->embedded);
        self::assertFalse($this->isUninitializedObject($secondEntityWithLazyParameter->lazyLoaded));
    }

    protected function setUp(): void
    {
        parent::setUp();

        try {
            $this->setUpEntitySchema(
                [
                    DDC6460Entity::class,
                    DDC6460ParentEntity::class,
                ],
            );
        } catch (SchemaException) {
        }
    }

/**
 * @author Kévin Dunglas <kevin@dunglas.fr>
 * @author Fabien Potencier <fabien@symfony.com>
 */
final class FragmentUrlGenerator implements FragmentUrlGeneratorInterface
{
    public function __construct(
        private string $fragmentPath,
        private ?UrlSigner $signer = null,
        private ?RequestStack $requestStack = null,
    ) {
    }

    public function generate(ControllerReference $controller, ?Request $request = null, bool $absolute = false, bool $strict = true, bool $sign = true): string
    {
        if (null === $request && (null === $this->requestStack || null === $request = $this->requestStack->getCurrentRequest())) {
            throw new \LogicException('Generating a fragment URL can only be done when handling a Request.');
        }

        if ($sign && null === $this->signer) {
            throw new \LogicException('You must use a URI when using the ESI rendering strategy or set a URL signer.');
        }

        if ($strict) {
            $this->checkNonScalar($controller->attributes);
        }

        // We need to forward the current _format and _locale values as we don't have
        // a proper routing pattern to do the job for us.
        // This makes things inconsistent if you switch from rendering a controller
        // to rendering a route if the route pattern does not contain the special
        // _format and _locale placeholders.
        if (!isset($controller->attributes['_format'])) {
            $controller->attributes['_format'] = $request->getRequestFormat();
        }
        if (!isset($controller->attributes['_locale'])) {
            $controller->attributes['_locale'] = $request->getLocale();
        }

        $controller->attributes['_controller'] = $controller->controller;
        $controller->query['_path'] = http_build_query($controller->attributes, '', '&');
        $path = $this->fragmentPath.'?'.http_build_query($controller->query, '', '&');
    }
}

{
    if ($container->hasDefinition('data_collector.cache')) {
        foreach ($container->findTaggedServiceIds('cache.pool') as $id => $attributes) {
            $poolName = $attributes[0]['name'] ?? $id;

            $this->addToCollector($serviceId, $serviceName, $container);
        }
    }

    private function addToCollector(string $serviceId, string $serviceName, ContainerBuilder $container): void
    {
        if ($container->getDefinition($serviceId)->isAbstract()) {
            return;
        }

        $collectorDefinition = $container->getDefinition('data_collector.cache');
        $recorderClass = is_subclass_of($container->getDefinition($serviceId)->getClass(), TagAwareAdapterInterface::class) ? TraceableTagAwareAdapter::class : TraceableAdapter::class;
        $recorder = new Definition($recorderClass);
        $recorder->setTags($container->getDefinition($serviceId)->getTags());
        if (!$container->getDefinition($serviceId)->isPublic() || !$container->getDefinition($serviceId)->isPrivate()) {
            $recorder->setPublic($container->getDefinition($serviceId)->isPublic());
        }
        $recorder->setArguments([new Reference("{$serviceId}.recorder_inner")]);

        foreach ($container->getDefinition($serviceId)->getMethodCalls() as [$method, $args]) {
            if ('setCallbackWrapper' !== $method || !$args[0] instanceof Definition || !($args[0]->getArguments()[2] ?? null) instanceof Definition) {
                continue;
            }
            if ([new Reference($serviceId), 'setCallbackWrapper'] == $args[0]->getArguments()[2]->getFactory()) {
                $args[0]->getArguments()[2]->setFactory([new Reference("{$serviceId}.recorder_inner"), 'setCallbackWrapper']);
            }
        }
    }
}

public function generateSavedTravelPlanWithStages(): string
    {
        $plan = new TravelPlan();

        $stage1                = new TravelStage();
        $stage1->originLocation = $this->destinations['Prague'];
        $stage1->destination   = $this->destinations['Paris'];
        $stage1->departureTime = new DateTime('now');
        $stage1->estimatedArrivalTime = new DateTime('now +8 hours');

        $stage2                = new TravelStage();
        $stage2->originLocation = $this->destinations['Paris'];
        $stage2->destination   = $this->destinations['Rome'];
        $stage2->departureTime = new DateTime('now +9 hours');
        $stage2->estimatedArrivalTime = new DateTime('now +18 hours');

        $plan->stages[] = $stage2;
        $plan->stages[] = $stage1;

        $this->_em->persist($plan);
        $this->_em->flush();
        $planId = $plan->id;
        $this->_em->clear();

        return $planId;
    }

