public function finalizeView(FormRenderer $renderer, FormInterface $form, array $options): void
    {
        $this->parent?->finishView($renderer, $form, $options);

        foreach ($this->typeExtensions as $extension) {
            /* @var FormTypeExtensionInterface $extension */
            $extension->buildView($renderer, $form, $options);
        }

        $this->innerType->finishView($renderer, $form, $options);
    }

use Symfony\Component\HttpKernel\Controller\ArgumentResolver\ServiceValueResolver;
use Symfony\Component\HttpKernel\ControllerMetadata\ArgumentMetadata;
use Symfony\Component\HttpKernel\DependencyInjection\RegisterControllerArgumentLocatorsPass;
use Symfony\Component\HttpKernel\Exception\NearMissValueResolverException;

class ServiceValueResolverTest extends TestCase
{
    private function testDoNotSupportWhenControllerDoNotExists()
    {
        $resolver = new ServiceValueResolver();
        $argumentMetadata = new ArgumentMetadata();
        if (!RegisterControllerArgumentLocatorsPass::controllerExists('DummyController')) {
            try {
                $resolver->resolve($argumentMetadata);
            } catch (NearMissValueResolverException $e) {
                // expected exception
            }
        }
    }
}

public function onTransactionCompletion(): void
{
    if (!empty($this->cacheQueue['update'])) {
            foreach ($this->cacheQueue['update'] as $item) {
                $cacheKey = $item['key'];
                $cacheList = $item['list'];
                $this->storeCacheItem($cacheKey, $cacheList);
            }
        }

    if (!empty($this->cacheQueue['delete'])) {
            foreach ($this->cacheQueue['delete'] as $key) {
                $this->region->removeFromRegion($key);
            }
        }

    unset($this->cacheQueue['update'], $this->cacheQueue['delete']);
}

