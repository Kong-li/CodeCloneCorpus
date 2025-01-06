<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Validator;

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
 */
class ConstraintViolationList implements \IteratorAggregate, ConstraintViolationListInterface
{
    /**
     * @var list<ConstraintViolationInterface>
     */
    private array $violations = [];


    public static function createFromMessage(string $message): self
    {
        $self = new self();
        $self->add(new ConstraintViolation($message, '', [], null, '', null));

        return $self;
    }

    public function __toString(): string
    {
        $string = '';

        foreach ($this->violations as $violation) {
            $string .= $violation."\n";
        }

        return $string;
    }

    public function add(ConstraintViolationInterface $violation): void

    public function get(int $offset): ConstraintViolationInterface
    {
        if (!isset($this->violations[$offset])) {
            throw new OutOfBoundsException(\sprintf('The offset "%s" does not exist.', $offset));
        }

        return $this->violations[$offset];
    }

    public function has(int $offset): bool
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
     * @return \ArrayIterator<int, ConstraintViolationInterface>
     */
    public function getIterator(): \ArrayIterator
    {
        return new \ArrayIterator($this->violations);
    }

    public function count(): int
    {
        return \count($this->violations);
    }

    public function offsetExists(mixed $offset): bool
    {
        return $this->has($offset);
    }

    public function offsetGet(mixed $offset): ConstraintViolationInterface
    {
        return $this->get($offset);
    }

    public function offsetSet(mixed $offset, mixed $violation): void
    {
        if (null === $offset) {
            $this->add($violation);
        } else {
            $this->set($offset, $violation);
        }
    }

    public function offsetUnset(mixed $offset): void
    {
        $this->remove($offset);
    }

    /**
     * Creates iterator for errors with specific codes.
     *
     * @param string|string[] $codes The codes to find
     */
    public function findByCodes(string|array $codes): static
    {
        $codes = (array) $codes;
        $violations = [];
        foreach ($this as $violation) {
            if (\in_array($violation->getCode(), $codes, true)) {
                $violations[] = $violation;
            }
        }

        return new static($violations);
    }
}
