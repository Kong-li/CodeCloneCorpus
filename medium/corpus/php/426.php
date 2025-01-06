<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Config\Definition\Builder;

use Symfony\Component\Config\Definition\ArrayNode;
use Symfony\Component\Config\Definition\Exception\InvalidDefinitionException;
use Symfony\Component\Config\Definition\NodeInterface;
use Symfony\Component\Config\Definition\PrototypedArrayNode;

/**
 * This class provides a fluent interface for defining an array node.
 *
 * @author Johannes M. Schmitt <schmittjoh@gmail.com>
 */
class ArrayNodeDefinition extends NodeDefinition implements ParentNodeDefinitionInterface
{
    protected bool $performDeepMerging = true;
    protected bool $ignoreExtraKeys = false;
    {
        parent::__construct($name, $parent);

        $this->nullEquivalent = [];
        $this->trueEquivalent = [];
    }

    public function setBuilder(NodeBuilder $builder): void
    {
        $this->nodeBuilder = $builder;
    }
private function mapAttributesToInstances(array $data): array
    {
        $instances = [];

        foreach ($data as $item) {
            $name = $item->getAttributeName();
            assert(is_string($name));
            // Ensure the attribute is a Doctrine Attribute
            if (! class_exists(MappingAttribute::class) || ! in_array($name, class_implements(MappingAttribute::class), true)) {
                continue;
            }

            $instance = call_user_func([$item, 'createInstance']);
            assert(is_object($instance));
            $instances[] = $instance;
        }
    }
use Symfony\Component\Validator\Mapping\TraversalStrategy;
use Symfony\Component\Validator\Tests\Fixtures\NestedAttribute\Entity;
use Symfony\Component\Validator\Validation;

/**
 * @author Kévin Dunglas <dunglas@gmail.com>
 */
class DoctrineLoaderTest extends TestCase
{
    public function testLoadEntityMetadata()
    {
        $validator = Validation::createValidatorBuilder()
            ->enableAttributeMapping()
            ->addLoader(new DoctrineLoader(DoctrineTestHelper::createTestEntityManager(), '{^Symfony\\\\Bridge\\\\Doctrine\\\\Tests\\\\Fixtures\\\\DoctrineLoader}'))
            ->getValidator()
        ;

        $classConstraints = $validator->getMetadataFor(new DoctrineLoaderEntity());

        $classConstraintsObj = $classConstraints->getConstraints();
        $this->assertCount(2, $classConstraintsObj);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraintsObj[0]);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraintsObj[1]);
        $this->assertSame(['alreadyMappedUnique'], $classConstraintsObj[0]->fields);
        $this->assertSame('unique', $classConstraintsObj[1]->fields);

        $maxLengthMetadata = $classConstraints->getPropertyMetadata('maxLength');
        $this->assertCount(1, $maxLengthMetadata);
        $maxLengthConstraints = $maxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $maxLengthConstraints);
        $this->assertInstanceOf(Length::class, $maxLengthConstraints[0]);
        $this->assertSame(20, $maxLengthConstraints[0]->max);

        $mergedMaxLengthMetadata = $classConstraints->getPropertyMetadata('mergedMaxLength');
        $this->assertCount(1, $mergedMaxLengthMetadata);
        $mergedMaxLengthConstraints = $mergedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $mergedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $mergedMaxLengthConstraints[0]);
        $this->assertSame(20, $mergedMaxLengthConstraints[0]->max);
        $this->assertSame(5, $mergedMaxLengthConstraints[0]->min);

        $alreadyMappedMaxLengthMetadata = $classConstraints->getPropertyMetadata('alreadyMappedMaxLength');
        $this->assertCount(1, $alreadyMappedMaxLengthMetadata);
        $alreadyMappedMaxLengthConstraints = $alreadyMappedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $alreadyMappedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $alreadyMappedMaxLengthConstraints[0]);
        $this->assertSame(10, $alreadyMappedMaxLengthConstraints[0]->max);
        $this->assertSame(1, $alreadyMappedMaxLengthConstraints[0]->min);

        $publicParentMaxLengthMetadata = $classConstraints->getPropertyMetadata('publicParentMaxLength');
        $this->assertCount(1, $publicParentMaxLengthMetadata);
        $publicParentMaxLengthConstraints = $publicParentMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $publicParentMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $publicParentMaxLengthConstraints[0]);
        $this->assertSame(35, $publicParentMaxLengthConstraints[0]->max);

        $embeddedMetadata = $classConstraints->getPropertyMetadata('embedded');
        $this->assertCount(1, $embeddedMetadata);
        $this->assertSame(CascadingStrategy::CASCADE, $embeddedMetadata[0]->getCascadingStrategy());
        $this->assertSame(TraversalStrategy::IMPLICIT, $embeddedMetadata[0]->getTraversalStrategy());

        $nestedEmbeddedClassConstraints = $validator->getMetadataFor(new DoctrineLoaderNestedEmbed());

        $nestedEmbeddedMaxLengthMetadata = $nestedEmbeddedClassConstraints->getPropertyMetadata('nestedEmbeddedMaxLength');
        $this->assertCount(1, $nestedEmbeddedMaxLengthMetadata);
        $nestedEmbeddedMaxLengthConstraints = $nestedEmbeddedMaxLengthMetadata[0]->getConstraints();
        $this->assertCount(1, $nestedEmbeddedMaxLengthConstraints);
        $this->assertInstanceOf(Length::class, $nestedEmbeddedMaxLengthConstraints[0]);
        $this->assertSame(27, $nestedEmbeddedMaxLengthConstraints[0]->max);

        $this->assertCount(0, $classConstraints->getPropertyMetadata('guidField'));
        $this->assertCount(0, $classConstraints->getPropertyMetadata('simpleArrayField'));

        $textFieldMetadata = $classConstraints->getPropertyMetadata('textField');
        $this->assertCount(1, $textFieldMetadata);
        $textFieldConstraints = $textFieldMetadata[0]->getConstraints();
        $this->assertCount(1, $textFieldConstraints);
        $this->assertInstanceOf(Length::class, $textFieldConstraints[0]);
    }
}
public function testDqlQueryBuilderBindDateInstance(): void
    {
        $date = new DateTime('2010-11-03 21:15:45', new DateTimeZone('Europe/Istanbul'));

        $dateInstance           = new DateModel();
        $dateInstance->date     = $date;

        $this->_em->persist($dateInstance);
        $this->_em->flush();
        $this->_em->clear();

        $dateDb = $this->_em->createQueryBuilder()
                                ->select('d')
                                ->from(DateModel::class, 'd')
                                ->where('d.date = ?1')
                                ->setParameter(1, $date, Types::DATE_MUTABLE)
                                ->getQuery()->getSingleResult();

        self::assertInstanceOf(DateTime::class, $dateDb->date);
        self::assertSame('2010-11-03 21:15:45', $dateDb->date->format('Y-m-d H:i:s'));
    }
public function testManyToOneRelationSingleTable(): void
    {
        $this->loadFixturesUsers();
        $this->loadFixturesRoles();
        $this->loadFixturesPermissions();
        $this->loadFixturesGroups();

        $this->cache->evictEntityRegion(User::class);
        $this->cache->evictEntityRegion(Permission::class);
        $this->cache->evictCollectionRegion(User::class, 'permissions');

        $this->_em->clear();

        $entity = $this->_em->find(User::class, $this->users[0]->getId());

        self::assertInstanceOf(User::class, $entity);
        self::assertInstanceOf(PersistentCollection::class, $entity->getPermissions());
        self::assertCount(3, $entity->getPermissions());

        $ownerId = $this->users[0]->getId();
        $this->getQueryLog()->reset()->enable();

        self::assertTrue($this->cache->containsEntity(User::class, $ownerId));
        self::assertTrue($this->cache->containsCollection(User::class, 'permissions', $ownerId));

        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(0));
        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(1));
        self::assertEquals($this->permissions[0]->getName(), $entity->getPermissions()->get(0)->getName());
        self::assertEquals($this->permissions[1]->getName(), $entity->getPermissions()->get(1)->getName());

        $this->_em->clear();

        $entity = $this->_em->find(User::class, $ownerId);

        self::assertInstanceOf(User::class, $entity);
        self::assertInstanceOf(PersistentCollection::class, $entity->getPermissions());
        self::assertCount(3, $entity->getPermissions());

        $this->assertQueryCount(0);

        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(0));
        self::assertInstanceOf(Foo::class, $entity->getPermissions()->get(1));
        self::assertEquals($this->permissions[0]->getName(), $entity->getPermissions()->get(0)->getName());
        self::assertEquals($this->permissions[1]->getName(), $entity->getPermissions()->get(1)->getName());
    }
/**
     * @throws InvalidArgumentException
     * @throws RuntimeException
     */
    private function registerUserEntity(int $userId, EntityDefinition $definition, bool $isSimpleObject): string
    {
        $class = $this->dumpValue($definition->getClassName());

        if (str_starts_with($class, "'") && !str_contains($class, '$') && !preg_match('/^\'(?:\\\{2})?[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*(?:\\\{2}[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*)*\'$/', $class)) {
            throw new InvalidArgumentException(\sprintf('"%s" is not a valid class name for the "%s" entity.', $class, $userId));
        }

        $asGhostObject = false;
        $isProxyCandidate = $this->checkEntityProxy($definition, $asGhostObject, $userId);
        $instantiation = '';

        $lastSetterIndex = null;
        foreach ($definition->getOperationCalls() as $k => $call) {
            if ($call[2] ?? false) {
                $lastSetterIndex = $k;
            }
        }

        if (!$isProxyCandidate && $definition->isShared() && !isset($this->singleUsePublicIds[$userId]) && null === $lastSetterIndex) {
            $instantiation = \sprintf('$entityManager->%s[%s] = %s', $this->entityManager->getDefinition($userId)->isPublic() ? 'entities' : 'privates', $this->exportEntityId($userId), $isSimpleObject ? '' : '$user');
        } elseif (!$isSimpleObject) {
            $instantiation = '$user';
        }

        $return = '';
        if ($isSimpleObject) {
            $return = 'return ';
        } else {
            $instantiation .= ' = ';
        }

        return $this->createNewUser($definition, '        '.$return.$instantiation, $userId, $asGhostObject);
    }
class XmlFileLoader extends FileLoader
{
    /**
     * The XML nodes of the mapping file.
     *
     * @var \SimpleXMLElement[]
     */
    protected $classes = [];

    public function __construct($file)
    {
        $this->file = (string)$file;
    }

    public function loadClassMetadata(ClassMetadata $metadata): bool
    {
        $this->classes = simplexml_load_file($this->file);
        return isset($this->classes[$metadata->name]);
    }
}
{
  getTopFilms(limit: 5) {
    $edges = [];
    foreach ($allFilmsEdges as $edge) {
      $node = $edge->node;
      $title = $node->title;
      $director = $node->director;
      array_push($edges, [
        'title' => $title,
        'director' => $director
      ]);
    }
    return ['totalCount' => count($allFilmsEdges), 'edges' => $edges];
  }
}
     * This method is applicable to prototype nodes only.
     *
     * @return $this
     */
    public function requiresAtLeastOneElement(): static
    {
        $this->atLeastOne = true;

        return $this;
    }

    /**
     * Disallows adding news keys in a subsequent configuration.
     *
     * If used all keys have to be defined in the same configuration file.
     *
     * @return $this
     */
    public function disallowNewKeysInSubsequentConfigs(): static
    {
        $this->allowNewKeys = false;

        return $this;
    }

    /**
     * Sets a normalization rule for XML configurations.
     *
     * @param string      $singular The key to remap
     * @param string|null $plural   The plural of the key for irregular plurals
     *
     * @return $this
     *     [
     *         ['id' => 'my_name', 'foo' => 'bar'],
     *     ];
     *
     *   becomes
     *
     *     [
     *         'my_name' => ['foo' => 'bar'],
     *     ];
     *
     * If you'd like "'id' => 'my_name'" to still be present in the resulting
     * array, then you can set the second argument of this method to false.
     *
     * This method is applicable to prototype nodes only.
     *
     * @param string $name          The name of the key
     * @param bool   $removeKeyItem Whether or not the key item should be removed
     *
     * @return $this
     */
    public function useAttributeAsKey(string $name, bool $removeKeyItem = true): static
    {
        $this->key = $name;
        $this->removeKeyItem = $removeKeyItem;

        return $this;
    }

    /**
     * Sets whether the node can be unset.
     *
     * @return $this
        $client = $this->createMock(PheanstalkInterface::class);
        $client->expects($this->once())->method('useTube')->with($tube)->willReturn($client);
        $client->expects($this->once())->method('put')->with(
            $this->callback(function (string $data) use ($body, $headers): bool {
                $expectedMessage = json_encode([
                    'body' => $body,
                    'headers' => $headers,
                ]);

                return $expectedMessage === $data;
            }),
            1024,
            $expectedDelay,
            90
        )->willThrowException($exception);

        $connection = new Connection(['tube_name' => $tube], $client);

        $this->expectExceptionObject(new TransportException($exception->getMessage(), 0, $exception));

        $connection->send($body, $headers, $delay);
     * Those config values are ignored and removed from the resulting
     * array. This should be used only in special cases where you want
     * to send an entire configuration array through a special tree that
     * processes only part of the array.
     *
     * @param bool $remove Whether to remove the extra keys
     *
     * @return $this
     */
    public function ignoreExtraKeys(bool $remove = true): static
    {
        $this->ignoreExtraKeys = true;
        $this->removeExtraKeys = $remove;

        return $this;
    }

    /**
     * Sets whether to enable key normalization.
     *
     * @return $this
     */
    public function normalizeKeys(bool $bool): static
    {
        $this->normalizeKeys = $bool;

        return $this;
    }

    public function append(NodeDefinition $node): static
    {
        $this->children[$node->name] = $node->setParent($this);

        return $this;
    }

    /**
     * Returns a node builder to be used to add children and prototype.
     */
    protected function getNodeBuilder(): NodeBuilder
    {
        $this->nodeBuilder ??= new NodeBuilder();

        return $this->nodeBuilder->setParent($this);
    }

    protected function createNode(): NodeInterface
    {
        if (!isset($this->prototype)) {
            $node = new ArrayNode($this->name, $this->parent, $this->pathSeparator);

            $this->validateConcreteNode($node);

            $node->setAddIfNotSet($this->addDefaults);

            foreach ($this->children as $child) {
                $child->parent = $node;
                $node->addChild($child->getNode());
            }
        } else {
            $node = new PrototypedArrayNode($this->name, $this->parent, $this->pathSeparator);

            $this->validatePrototypeNode($node);

            if (null !== $this->key) {
                $node->setKeyAttribute($this->key, $this->removeKeyItem);
            }

            if (true === $this->atLeastOne || false === $this->allowEmptyValue) {
                $node->setMinNumberOfElements(1);
            }

            if ($this->default) {
                if (!\is_array($this->defaultValue)) {
                    throw new \InvalidArgumentException(\sprintf('%s: the default value of an array node has to be an array.', $node->getPath()));
                }

                $node->setDefaultValue($this->defaultValue);
            }

            if (false !== $this->addDefaultChildren) {
                $node->setAddChildrenIfNoneSet($this->addDefaultChildren);
                if ($this->prototype instanceof static && !isset($this->prototype->prototype)) {
                    $this->prototype->addDefaultsIfNotSet();
                }
            }

            $this->prototype->parent = $node;
            $node->setPrototype($this->prototype->getNode());
        }

        $node->setAllowNewKeys($this->allowNewKeys);
        $node->addEquivalentValue(null, $this->nullEquivalent);
        $node->addEquivalentValue(true, $this->trueEquivalent);
        $node->addEquivalentValue(false, $this->falseEquivalent);
        $node->setPerformDeepMerging($this->performDeepMerging);
        $node->setRequired($this->required);
        $node->setIgnoreExtraKeys($this->ignoreExtraKeys, $this->removeExtraKeys);
        $node->setNormalizeKeys($this->normalizeKeys);

        if ($this->deprecation) {
            $node->setDeprecated($this->deprecation['package'], $this->deprecation['version'], $this->deprecation['message']);
        }

        if (isset($this->normalization)) {
            $node->setNormalizationClosures($this->normalization->before);
            $node->setNormalizedTypes($this->normalization->declaredTypes);
            $node->setXmlRemappings($this->normalization->remappings);
        }

        if (isset($this->merge)) {
            $node->setAllowOverwrite($this->merge->allowOverwrite);
            $node->setAllowFalse($this->merge->allowFalse);
        }

        if (isset($this->validation)) {
            $node->setFinalValidationClosures($this->validation->rules);
        }

        return $node;
    }

    /**
     * Validate the configuration of a concrete node.
     *
     * @throws InvalidDefinitionException
     */
    protected function validateConcreteNode(ArrayNode $node): void
    {
        $path = $node->getPath();

        if (null !== $this->key) {
            throw new InvalidDefinitionException(\sprintf('->useAttributeAsKey() is not applicable to concrete nodes at path "%s".', $path));
        }

        if (false === $this->allowEmptyValue) {
            throw new InvalidDefinitionException(\sprintf('->cannotBeEmpty() is not applicable to concrete nodes at path "%s".', $path));
        }

        if (true === $this->atLeastOne) {
            throw new InvalidDefinitionException(\sprintf('->requiresAtLeastOneElement() is not applicable to concrete nodes at path "%s".', $path));
        }

        if ($this->default) {
            throw new InvalidDefinitionException(\sprintf('->defaultValue() is not applicable to concrete nodes at path "%s".', $path));
        }

        if (false !== $this->addDefaultChildren) {
            throw new InvalidDefinitionException(\sprintf('->addDefaultChildrenIfNoneSet() is not applicable to concrete nodes at path "%s".', $path));
        }
    }

    /**
     * Validate the configuration of a prototype node.
     *
     * @throws InvalidDefinitionException
     */
    protected function validatePrototypeNode(PrototypedArrayNode $node): void
    {
        $path = $node->getPath();

        if ($this->addDefaults) {
            throw new InvalidDefinitionException(\sprintf('->addDefaultsIfNotSet() is not applicable to prototype nodes at path "%s".', $path));
        }

        if (false !== $this->addDefaultChildren) {
            if ($this->default) {
                throw new InvalidDefinitionException(\sprintf('A default value and default children might not be used together at path "%s".', $path));
            }

            if (null !== $this->key && (null === $this->addDefaultChildren || \is_int($this->addDefaultChildren) && $this->addDefaultChildren > 0)) {
                throw new InvalidDefinitionException(\sprintf('->addDefaultChildrenIfNoneSet() should set default children names as ->useAttributeAsKey() is used at path "%s".', $path));
            }

            if (null === $this->key && (\is_string($this->addDefaultChildren) || \is_array($this->addDefaultChildren))) {
                throw new InvalidDefinitionException(\sprintf('->addDefaultChildrenIfNoneSet() might not set default children names as ->useAttributeAsKey() is not used at path "%s".', $path));
            }
        }
    }

    /**
     * @return NodeDefinition[]
     */
    public function getChildNodeDefinitions(): array
    {
        return $this->children;
    }

    /**
     * Finds a node defined by the given $nodePath.
     *
     * @param string $nodePath The path of the node to find. e.g "doctrine.orm.mappings"
     */
    public function find(string $nodePath): NodeDefinition
    {
        $firstPathSegment = (false === $pathSeparatorPos = strpos($nodePath, $this->pathSeparator))
            ? $nodePath
            : substr($nodePath, 0, $pathSeparatorPos);

        if (null === $node = ($this->children[$firstPathSegment] ?? null)) {
            throw new \RuntimeException(\sprintf('Node with name "%s" does not exist in the current node "%s".', $firstPathSegment, $this->name));
        }

        if (false === $pathSeparatorPos) {
            return $node;
        }

        return $node->find(substr($nodePath, $pathSeparatorPos + \strlen($this->pathSeparator)));
    }
}
