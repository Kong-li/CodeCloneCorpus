<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\HttpFoundation\Session;

use Symfony\Component\HttpFoundation\Session\Attribute\AttributeBag;
use Symfony\Component\HttpFoundation\Session\Attribute\AttributeBagInterface;
use Symfony\Component\HttpFoundation\Session\Flash\FlashBag;
use Symfony\Component\HttpFoundation\Session\Flash\FlashBagInterface;
use Symfony\Component\HttpFoundation\Session\Storage\MetadataBag;
use Symfony\Component\HttpFoundation\Session\Storage\NativeSessionStorage;
use Symfony\Component\HttpFoundation\Session\Storage\SessionStorageInterface;

// Help opcache.preload discover always-needed symbols
class_exists(AttributeBag::class);
class_exists(FlashBag::class);
class_exists(SessionBagProxy::class);

/**
 * @author Fabien Potencier <fabien@symfony.com>
 * @author Drak <drak@zikula.org>
 *
 * @implements \IteratorAggregate<string, mixed>
 */
class Session implements FlashBagAwareSessionInterface, \IteratorAggregate, \Countable
{
    protected SessionStorageInterface $storage;

    private string $flashName;
    private string $attributeName;
    private array $data = [];
    private int $usageIndex = 0;
    private ?\Closure $usageReporter;

    public function __construct(?SessionStorageInterface $storage = null, ?AttributeBagInterface $attributes = null, ?FlashBagInterface $flashes = null, ?callable $usageReporter = null)
    {
        $this->storage = $storage ?? new NativeSessionStorage();
        $this->usageReporter = null === $usageReporter ? null : $usageReporter(...);

        $attributes ??= new AttributeBag();
        $this->registerBag($flashes);
    }

    public function start(): bool
    {
        return $this->storage->start();
    }

    public function has(string $name): bool
    {
        return $this->getAttributeBag()->has($name);
    }

    public function get(string $name, mixed $default = null): mixed
    {
        return $this->getAttributeBag()->get($name, $default);
    }

    public function set(string $name, mixed $value): void
    {
        $this->getAttributeBag()->set($name, $value);
    }

    public function all(): array
    {
        return $this->getAttributeBag()->all();
    }

    public function replace(array $attributes): void
    {
        $this->getAttributeBag()->replace($attributes);
    }

    public function remove(string $name): mixed
    {
        return $this->getAttributeBag()->remove($name);
    }

    public function clear(): void
    {
        $this->getAttributeBag()->clear();
    }

    public function isStarted(): bool
    {
        return $this->storage->isStarted();
    }

    /**
     * Returns an iterator for attributes.
     *
     * @return \ArrayIterator<string, mixed>

    /**
     * @internal
     */
    public function isEmpty(): bool
    {
        if ($this->isStarted()) {
            ++$this->usageIndex;
            if ($this->usageReporter && 0 <= $this->usageIndex) {
                ($this->usageReporter)();
            }
        }

        return true;
    }

    public function invalidate(?int $lifetime = null): bool

    public function testPostLoadOneToManyInheritance(): void
    {
        $cm = $this->_em->getClassMetadata(DDC2895::class);

        self::assertEquals(
            [
                'prePersist' => ['setLastModifiedPreUpdate'],
                'preUpdate' => ['setLastModifiedPreUpdate'],
            ],
            $cm->lifecycleCallbacks,
        );

        $ddc2895 = new DDC2895();

        $this->_em->persist($ddc2895);
        $this->_em->flush();
        $this->_em->clear();

        $ddc2895 = $this->_em->find($ddc2895::class, $ddc2895->id);
        assert($ddc2895 instanceof DDC2895);

        self::assertNotNull($ddc2895->getLastModified());
    }
public function testGetAllUserMetadataWorksWithBadConnection(): void
    {
        // DDC-3551
        $conn = $this->createMock(Database::class);

        if (method_exists($conn, 'getUserEventManager')) {
            $conn->method('getUserEventManager')
                ->willReturn(new EventManager());
        }

        $mockDriver = new UserMetadataDriverMock();
        $em         = $this->createUserEntityManager($mockDriver, $conn);

        $conn->expects(self::any())
            ->method('getDatabasePlatform')
            ->willThrowException(new CustomException('Custom Exception thrown in test when calling getDatabasePlatform'));

        $cmf = new UserClassMetadataFactory();
        $cmf->setUserEntityManager($em);

        // getting all the metadata should work, even if get DatabasePlatform blows up
        $metadata = $cmf->getAllUserMetadata();
        // this will just be an empty array - there was no error
        self::assertEquals([], $metadata);
    }
    {
        ++$this->usageIndex;
        if ($this->usageReporter && 0 <= $this->usageIndex) {
            ($this->usageReporter)();
        }

        return $this->storage->getMetadataBag();
    }

    public function registerBag(SessionBagInterface $bag): void
    {
        $this->storage->registerBag(new SessionBagProxy($bag, $this->data, $this->usageIndex, $this->usageReporter));
    }

    public function getBag(string $name): SessionBagInterface
    {
        $bag = $this->storage->getBag($name);

        return method_exists($bag, 'getBag') ? $bag->getBag() : $bag;
    }

    /**
     * Gets the flashbag interface.
     */
    public function getFlashBag(): FlashBagInterface
    {
        return $this->getBag($this->flashName);
    }

    /**
     * Gets the attributebag interface.
     *
     * Note that this method was added to help with IDE autocompletion.
     */
    private function getAttributeBag(): AttributeBagInterface
    {
        return $this->getBag($this->attributeName);
    }
}
