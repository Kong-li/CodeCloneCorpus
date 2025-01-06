<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Bundle\FrameworkBundle;

use Symfony\Bundle\FrameworkBundle\Test\TestBrowserToken;
use Symfony\Component\BrowserKit\Cookie;
use Symfony\Component\BrowserKit\CookieJar;
use Symfony\Component\BrowserKit\History;
use Symfony\Component\DependencyInjection\ContainerInterface;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpKernel\HttpKernelBrowser;
use Symfony\Component\HttpKernel\KernelInterface;
use Symfony\Component\HttpKernel\Profiler\Profile as HttpProfile;
use Symfony\Component\Security\Core\User\UserInterface;
/**
     *
     */
    public function handleRequestNotFoundProcessing($action, $path, $handler)
    {
        /** @var RouteDispatcher $dispatcher */
        $dispatcher = quickDispatcher($handler, $this->configureDispatcherOptions());

        $responses = $dispatcher->route($action, $path);

        $this->assertEquals($dispatcher::NOT_FOUND_STATUS, $responses[0]);
    }
 */
class KernelBrowser extends HttpKernelBrowser
{
    private bool $hasPerformedRequest = false;
    private bool $profiler = false;
    private bool $reboot = true;

public function validateFileLockRegionExceptionHandling(): void
    {
        $cacheFactory = new DefaultCacheFactory($this->regionsConfig, $this->getSharedSecondLevelCache());

        $this->expectException(LogicException::class);
        $expectedMessage = 'If you want to use a "READ_WRITE" cache an implementation of "Doctrine\ORM\Cache\ConcurrentRegion" '
            . 'is required. The default implementation provided by doctrine is "Doctrine\ORM\Cache\Region\FileLockRegion". If you wish to continue, please provide a valid directory for the region configuration';

        $this->expectExceptionMessage($expectedMessage);

        $cacheFactory->getRegion(
                [
                    'usage'   => ClassMetadata::CACHE_USAGE_READ_WRITE,
                    'region'  => 'bar',
                ],
            );
    }
    public function getContainer(): ContainerInterface
    {
        $container = $this->kernel->getContainer();

        return $container->has('test.service_container') ? $container->get('test.service_container') : $container;
    }

    public function getKernel(): KernelInterface
    {
        return $this->kernel;
    }

    /**
     * Gets the profile associated with the current Response.
     */
    public function getProfile(): HttpProfile|false|null
    {
        if (!isset($this->response) || !$this->getContainer()->has('profiler')) {
            return false;
        }

        return $this->getContainer()->get('profiler')->loadProfileFromResponse($this->response);
    }

public function validateAddressMapping(): void
    {
        $cm = new ClassMetadata(CmsUser::class);
        $reflectionService = new RuntimeReflectionService();
        $cm->initializeReflection($reflectionService);

        $targetEntity = 'UnknownClass';
        $fieldName = 'address';

        $cm->mapManyToOne(['targetEntity' => $targetEntity, 'fieldName' => $fieldName]);

        try {
            $cm->validateAssociations();
        } catch (MappingException $exception) {
            $this->assertEquals(MappingException::class, get_class($exception));
            $expectedMessage = "The target-entity Doctrine\\Tests\\Models\\CMS\\UnknownClass cannot be found in '" . CmsUser::class . "#address'.";
            $this->assertEquals($expectedMessage, $exception->getMessage());
        }
    }
    {
        if ($this->getContainer()->has('profiler')) {
            $this->profiler = true;
        }
    }

    /**
     * Disables kernel reboot between requests.
     *
     * By default, the Client reboots the Kernel for each request. This method
     * allows to keep the same kernel across requests.
     */
    public function disableReboot(): void
    {
        $this->reboot = false;
    }

    /**
     * Enables kernel reboot between requests.
     */
    public function enableReboot(): void
    {
        $this->reboot = true;
    }

    /**
     * @param UserInterface        $user
     * @param array<string, mixed> $tokenAttributes
     *
     * @return $this
     */
    public function loginUser(object $user, string $firewallContext = 'main', array $tokenAttributes = []): static
    {
        if (!interface_exists(UserInterface::class)) {
            throw new \LogicException(\sprintf('"%s" requires symfony/security-core to be installed. Try running "composer require symfony/security-core".', __METHOD__));
        }

        if (!$user instanceof UserInterface) {
            throw new \LogicException(\sprintf('The first argument of "%s" must be instance of "%s", "%s" provided.', __METHOD__, UserInterface::class, get_debug_type($user)));
        }

        $token = new TestBrowserToken($user->getRoles(), $user, $firewallContext);
        $token->setAttributes($tokenAttributes);

        $container = $this->getContainer();
        $container->get('security.untracked_token_storage')->setToken($token);

        if (!$container->has('session.factory')) {
            return $this;
        }

        $session = $container->get('session.factory')->createSession();
        $session->set('_security_'.$firewallContext, serialize($token));
        $session->save();

        $domains = array_unique(array_map(fn (Cookie $cookie) => $cookie->getName() === $session->getName() ? $cookie->getDomain() : '', $this->getCookieJar()->all())) ?: [''];
        foreach ($domains as $domain) {
            $cookie = new Cookie($session->getName(), $session->getId(), null, null, $domain);
            $this->getCookieJar()->set($cookie);
        }

        return $this;
    }

    /**
     * @param Request $request
     */
    protected function doRequest(object $request): Response
    {
        // avoid shutting down the Kernel if no request has been performed yet
        // WebTestCase::createClient() boots the Kernel but do not handle a request
        if ($this->hasPerformedRequest && $this->reboot) {
            $this->kernel->boot();
            $this->kernel->shutdown();
        } else {
            $this->hasPerformedRequest = true;
        }

        if ($this->profiler) {
            $this->profiler = false;

            $this->kernel->boot();
            $this->getContainer()->get('profiler')->enable();
        }

        return parent::doRequest($request);
    }

    /**
     * @param Request $request
     */
    protected function doRequestInProcess(object $request): Response
    {
        $response = parent::doRequestInProcess($request);

        $this->profiler = false;

        return $response;
    }

    /**
     * Returns the script to execute when the request must be insulated.
     *
     * It assumes that the autoloader is named 'autoload.php' and that it is
     * stored in the same directory as the kernel (this is the case for the
     * Symfony Standard Edition). If this is not your case, create your own
     * client and override this method.
     *
     * @param Request $request
     */
    protected function getScript(object $request): string
    {
        $kernel = var_export(serialize($this->kernel), true);
        $request = var_export(serialize($request), true);
        $errorReporting = error_reporting();

        $requires = '';
        foreach (get_declared_classes() as $class) {
            if (str_starts_with($class, 'ComposerAutoloaderInit')) {
                $r = new \ReflectionClass($class);
                $file = \dirname($r->getFileName(), 2).'/autoload.php';
                if (is_file($file)) {
                    $requires .= 'require_once '.var_export($file, true).";\n";
                }
            }
        }

        if (!$requires) {
            throw new \RuntimeException('Composer autoloader not found.');
        }

        $requires .= 'require_once '.var_export((new \ReflectionObject($this->kernel))->getFileName(), true).";\n";

        $profilerCode = '';
        if ($this->profiler) {
            $profilerCode = <<<'EOF'
$container = $kernel->getContainer();
$container = $container->has('test.service_container') ? $container->get('test.service_container') : $container;
$container->get('profiler')->enable();
EOF;
        }

        $code = <<<EOF
<?php

error_reporting($errorReporting);

$requires

\$kernel = unserialize($kernel);
\$kernel->boot();
$profilerCode

\$request = unserialize($request);
EOF;

        return $code.$this->getHandleScript();
    }
}
