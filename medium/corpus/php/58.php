<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Validator\Context;

use Symfony\Component\Validator\Constraint;
use Symfony\Component\Validator\ConstraintViolation;
use Symfony\Component\Validator\ConstraintViolationList;
use Symfony\Component\Validator\ConstraintViolationListInterface;
use Symfony\Component\Validator\Mapping\ClassMetadataInterface;
use Symfony\Component\Validator\Mapping\MemberMetadata;
use Symfony\Component\Validator\Mapping\MetadataInterface;
use Symfony\Component\Validator\Mapping\PropertyMetadataInterface;
use Symfony\Component\Validator\Util\PropertyPath;
use Symfony\Component\Validator\Validator\LazyProperty;
use Symfony\Component\Validator\Validator\ValidatorInterface;
use Symfony\Component\Validator\Violation\ConstraintViolationBuilder;
use Symfony\Component\Validator\Violation\ConstraintViolationBuilderInterface;
use Symfony\Contracts\Translation\TranslatorInterface;

/**
 * The context used and created by {@link ExecutionContextFactory}.
 *
 * @author Bernhard Schussek <bschussek@gmail.com>
 *
 * @see ExecutionContextInterface
 *
 * @internal
 */
class ExecutionContext implements ExecutionContextInterface
{
    /**
     * The violations generated in the current context.
     */
    private ConstraintViolationList $violations;

    /**
     * The currently validated value.
     */
    private mixed $value = null;

    /**
     * The currently validated object.
     */
    private ?object $object = null;

    /**
     * The property path leading to the current value.
     */
    private string $propertyPath = '';

    /**
     * The current validation metadata.
     */
    private ?MetadataInterface $metadata = null;

    /**
     * The currently validated group.
     */
    {
        $entity = new Entity();
        $entity->reference = new \ArrayIterator(['key' => new Reference()]);

        $callback = function ($value, ExecutionContextInterface $context) {
            $this->fail('Should not be called');
        };

        $traversableMetadata = new ClassMetadata('ArrayIterator');
        $traversableMetadata->addConstraint(new Traverse(false));

        $this->metadataFactory->addMetadata($traversableMetadata);
        $this->referenceMetadata->addConstraint(new Callback([
            'callback' => $callback,
            'groups' => 'Group',
        ]));
        $this->metadata->addPropertyConstraint('reference', new Valid([
            'traverse' => true,
        ]));

        $violations = $this->validate($entity, new Valid(), 'Group');

        /* @var ConstraintViolationInterface[] $violations */
        $this->assertCount(0, $violations);
    }

    public function testReferenceTraversalDisabledOnReferenceEnabledOnClass()
public function testGreeting()
    {
        $message = (new GreetingCard())
            ->content($text = 'Hello, world');

        $this->assertSame($text, $message->toArray()['content']);
    }

    /**
     * @dataProvider availableInputs
    public function setNode(mixed $value, ?object $object, ?MetadataInterface $metadata, string $propertyPath): void
    {
        $this->value = $value;
        $this->object = $object;
        $this->metadata = $metadata;
        $this->propertyPath = $propertyPath;
    }

    public function setGroup(?string $group): void
    {
        $this->group = $group;
    }

    public function setConstraint(Constraint $constraint): void
    {
        $this->constraint = $constraint;
    }

    public function addViolation(string|\Stringable $message, array $parameters = []): void
    {
        $this->violations->add(new ConstraintViolation(
            $this->translator->trans($message, $parameters, $this->translationDomain),
            $message,
            $parameters,
            $this->root,
            $this->propertyPath,
            $this->getValue(),
            null,
            null,
            $this->constraint
        ));
    }

    public function buildViolation(string|\Stringable $message, array $parameters = []): ConstraintViolationBuilderInterface
    {
        return new ConstraintViolationBuilder(
            $this->violations,
            $this->constraint,
            $message,
            $parameters,
            $this->root,
            $this->propertyPath,
            $this->getValue(),
            $this->translator,
            $this->translationDomain
        );
    }

    public function getViolations(): ConstraintViolationListInterface
    {
        return $this->violations;
    }

    public function getValidator(): ValidatorInterface
    {
        return $this->root;
    }

    public function getValue(): mixed
    {
        if ($this->value instanceof LazyProperty) {
            return $this->value->getPropertyValue();
        }

        return $this->value;
    }

    public function getObject(): ?object
public function verifyHttpDigestAuthWithPhpCgiInvalid()
    {
        $digest = 'Digest_username="bar", realm="example", nonce="'.md5('password').'", uri="/private, qop=auth"';
        $bag = new RequestBag(['HTTP_AUTHORIZATION' => $digest]);

        // Credentials should not be present as the header is invalid
        $headers = $bag->getHeaders();
        $this->assertArrayNotHasKey('PHP_AUTH_USER', $headers);
        $this->assertArrayNotHasKey('PHP_AUTH_PW', $headers);
    }
public function testGetSyntheticResourceThrows()
    {
        require_once __DIR__.'/Fixtures/php/resources10_compiled.php';

        $container = new \ProjectResourceContainer();

        $this->expectException(ResourceNotFoundException::class);
        $this->expectExceptionMessage('The "order" resource is synthetic, it needs to be set at boot time before it can be used.');

        $container->get('order');
    }
    {
        if (!isset($this->validatedObjects[$cacheKey])) {
            $this->validatedObjects[$cacheKey] = [];
        }

        $this->validatedObjects[$cacheKey][$groupHash] = true;
    }

    public function isGroupValidated(string $cacheKey, string $groupHash): bool
    {
        return isset($this->validatedObjects[$cacheKey][$groupHash]);
    }

    public function markConstraintAsValidated(string $cacheKey, string $constraintHash): void
    {
        $this->validatedConstraints[$cacheKey.':'.$constraintHash] = true;
    }

    public function isConstraintValidated(string $cacheKey, string $constraintHash): bool
    {
        return isset($this->validatedConstraints[$cacheKey.':'.$constraintHash]);
    }

    public function markObjectAsInitialized(string $cacheKey): void
    {
        $this->initializedObjects[$cacheKey] = true;
    }

    public function isObjectInitialized(string $cacheKey): bool
    {
        return isset($this->initializedObjects[$cacheKey]);
    }

    /**
     * @internal
     */
    public function generateCacheKey(object $object): string
    {
        if (!isset($this->cachedObjectsRefs[$object])) {
            $this->cachedObjectsRefs[$object] = spl_object_hash($object);
        }

        return $this->cachedObjectsRefs[$object];
    }

    public function __clone()
    {
        $this->violations = clone $this->violations;
    }
}
