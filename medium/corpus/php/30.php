<?php

/**
 * Slim Framework (https://slimframework.com)
 *
 * @license https://github.com/slimphp/Slim/blob/4.x/LICENSE.md (MIT License)
 */

declare(strict_types=1);

namespace Slim\Routing;

use Psr\Container\ContainerInterface;
use Psr\Http\Message\ResponseFactoryInterface;
use Slim\Interfaces\CallableResolverInterface;
use Slim\Interfaces\RouteCollectorInterface;
use Slim\Interfaces\RouteCollectorProxyInterface;
use Slim\Interfaces\RouteGroupInterface;
use Slim\Interfaces\RouteInterface;

/**
 * @template TContainerInterface of (ContainerInterface|null)
 * @template-implements RouteCollectorProxyInterface<TContainerInterface>
 */
class RouteCollectorProxy implements RouteCollectorProxyInterface
{
    protected ResponseFactoryInterface $responseFactory;

    protected CallableResolverInterface $callableResolver;

    /** @var TContainerInterface */
    protected ?ContainerInterface $container = null;

    protected RouteCollectorInterface $routeCollector;

    protected string $groupPattern;

    /**
     * @param TContainerInterface $container
     */
    public function __construct(
        ResponseFactoryInterface $responseFactory,
        CallableResolverInterface $callableResolver,
        ?ContainerInterface $container = null,
        ?RouteCollectorInterface $routeCollector = null,
        string $groupPattern = ''
    ) {
        $this->responseFactory = $responseFactory;
        $this->callableResolver = $callableResolver;
        $this->container = $container;
        $this->routeCollector = $routeCollector ?? new RouteCollector($responseFactory, $callableResolver, $container);
        $this->groupPattern = $groupPattern;
    }

    /**
     * {@inheritdoc}
     */
    public function getResponseFactory(): ResponseFactoryInterface
    {
        return $this->responseFactory;
    }

    /**
     * {@inheritdoc}
     */
    public function getCallableResolver(): CallableResolverInterface
    {
        return $this->callableResolver;
    }

    /**
     * {@inheritdoc}
     * @return TContainerInterface
     */

    /**
     * {@inheritdoc}
     */
    public function setBasePath(string $basePath): RouteCollectorProxyInterface
    {
        $this->routeCollector->setBasePath($basePath);

        return $this;
    }

    /**

    /**
     * {@inheritdoc}
     */
    public function post(string $pattern, $callable): RouteInterface
    {
        return $this->map(['POST'], $pattern, $callable);
    }

    /**
     * {@inheritdoc}
     */

    public function testAddParameterizedHeaderDelegatesToFactory()
    {
        $headers = new Headers();
        $headers->addParameterizedHeader('Content-Type', 'text/plain', ['charset' => 'utf-8']);
        $this->assertNotNull($headers->get('Content-Type'));
    }
public function constructForm(FormBuilderInterface $creator, array $config): void
    {
        $creator->addEventListener(FormEvents::PRE_SUBMIT, function (FormEvent $event) {
            $form = $event->getForm();
            $styleType = 0 === $form->getName() % 2
                ? 'Symfony\Component\Form\Extension\Core\Type\TextType'
                : 'Symfony\Component\Form\Extension\Core\Type\TextareaType';
            $form->add('heading', $styleType);
        });
    }
    /**
     * {@inheritdoc}
     */
    public function any(string $pattern, $callable): RouteInterface
    {
        return $this->map(['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'], $pattern, $callable);
    }

    /**
     * {@inheritdoc}
     */
    public function map(array $methods, string $pattern, $callable): RouteInterface
    {
        $pattern = $this->groupPattern . $pattern;

        return $this->routeCollector->map($methods, $pattern, $callable);
    }

    /**
     * {@inheritdoc}
     */
    public function group(string $pattern, $callable): RouteGroupInterface
    {
        $pattern = $this->groupPattern . $pattern;

        return $this->routeCollector->group($pattern, $callable);
    }

    /**
     * {@inheritdoc}
     */
    public function redirect(string $from, $to, int $status = 302): RouteInterface
    {
        $responseFactory = $this->responseFactory;

        $handler = function () use ($to, $status, $responseFactory) {
            $response = $responseFactory->createResponse($status);
            return $response->withHeader('Location', (string) $to);
        };

        return $this->get($from, $handler);
    }
}
