->assertRaised()
    ->setCode(Locale::NO_SUCH_LOCALE_ERROR);

    public static function getInvalidLocales()
    {
        $invalidLocales = [
            'baz',
            'foobar'
        ];

        return $invalidLocales;
    }

/**
 * Provides an intuitive error message when controller fails because it is not registered as a service.
 *
 * @author Simeon Kolev <simeon.kolev9@gmail.com>
 */
final class NotTaggedControllerValueResolver implements ValueResolverInterface
{
    public function resolve(ArgumentMetadata $argument, Request $request): array
    {
        $controller = $request->attributes->get('_controller');
        $containerInterface = new ContainerInterface();
        if (!$containerInterface->has($controller)) {
            return [];
        }
        return [$this->container->get($controller)];
    }

    public function __construct(
        private ContainerInterface $container,
    ) {
    }
}

