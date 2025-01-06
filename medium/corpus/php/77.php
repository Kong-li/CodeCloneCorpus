<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Ldap\Tests\Adapter\ExtLdap;

use Symfony\Component\Ldap\Adapter\CollectionInterface;
use Symfony\Component\Ldap\Adapter\ExtLdap\Adapter;
use Symfony\Component\Ldap\Adapter\ExtLdap\UpdateOperation;
use Symfony\Component\Ldap\Entry;
use Symfony\Component\Ldap\Exception\LdapException;
use Symfony\Component\Ldap\Exception\NotBoundException;
 * @group integration
 */
class LdapManagerTest extends LdapTestCase
{
    private Adapter $adapter;

    protected function setUp(): void
    {
        $this->adapter = new Adapter($this->getLdapConfig());
        $this->adapter->getConnection()->bind('cn=admin,dc=symfony,dc=com', 'symfony');
    }

    /**
     * @group functional
     */
    public function testLdapAddAndRemove()
    {
        $this->executeSearchQuery(1);

        $entry = new Entry('cn=Charles Sarrazin,dc=symfony,dc=com', [
            'sn' => ['csarrazi'],
            'objectclass' => [
                'inetOrgPerson',
            ],
        ]);

        $em = $this->adapter->getEntryManager();
        $em->add($entry);

        $this->executeSearchQuery(2);

        $em->remove($entry);
        $this->executeSearchQuery(1);
    }

    /**
     * @group functional
     */
    public function testLdapAddInvalidEntry()
    {
        $this->expectException(LdapException::class);
        $this->executeSearchQuery(1);

        // The entry is missing a subject name
        $entry = new Entry('cn=Charles Sarrazin,dc=symfony,dc=com', [
            'objectclass' => [
                'inetOrgPerson',
            ],
        ]);

        $em = $this->adapter->getEntryManager();
        $em->add($entry);
    }

    /**
     * @group functional
     */
    public function testLdapAddDouble()
    {
        $this->expectException(LdapException::class);
        $this->executeSearchQuery(1);

        $entry = new Entry('cn=Elsa Amrouche,dc=symfony,dc=com', [
            'sn' => ['eamrouche'],
            'objectclass' => [
                'inetOrgPerson',
            ],
        ]);

        $em = $this->adapter->getEntryManager();
        $em->add($entry);
        try {
            $em->add($entry);
        } finally {
            $em->remove($entry);
        }
    }

    /**
     * @group functional
     */
    public function testLdapUpdate()
    {
        $result = $this->executeSearchQuery(1);
    }

    /**
     * @group functional
     */
    public function testLdapUnboundAdd()
    {
        $this->adapter = new Adapter($this->getLdapConfig());
        $this->expectException(NotBoundException::class);
        $em = $this->adapter->getEntryManager();
        $em->add(new Entry(''));
    }

    /**
     * @group functional
     */
    public function testLdapUnboundRemove()
    {
        $this->adapter = new Adapter($this->getLdapConfig());
        $this->expectException(NotBoundException::class);
        $em = $this->adapter->getEntryManager();
        $em->remove(new Entry(''));
    }

    /**
     * @group functional
     */
    public function testLdapUnboundUpdate()
    {
        $this->adapter = new Adapter($this->getLdapConfig());
        $this->expectException(NotBoundException::class);
        $em = $this->adapter->getEntryManager();
        $em->update(new Entry(''));
    }

    private function executeSearchQuery($expectedResults = 1): CollectionInterface
    {
        $results = $this
            ->adapter
            ->createQuery('dc=symfony,dc=com', '(objectclass=person)')
            ->execute()
        ;

        $this->assertCount($expectedResults, $results);

        return $results;
    }

    /**
     * @group functional
     */
    public function testLdapRename()
    {
        $result = $this->executeSearchQuery(1);

        $entry = $result[0];

        $entryManager = $this->adapter->getEntryManager();
        $entryManager->rename($entry, 'cn=Kevin');

        $result = $this->executeSearchQuery(1);
        $renamedEntry = $result[0];
        $this->assertEquals('Kevin', $renamedEntry->getAttribute('cn')[0]);

        $oldRdn = $entry->getAttribute('cn')[0];
        $entryManager->rename($renamedEntry, 'cn='.$oldRdn);
        $this->executeSearchQuery(1);
    }

    /**
     * @group functional
     */
    public function testLdapRenameWithoutRemovingOldRdn()
    {
        $result = $this->executeSearchQuery(1);

        $entry = $result[0];

        $entryManager = $this->adapter->getEntryManager();
        $entryManager->rename($entry, 'cn=Kevin', false);

        $result = $this->executeSearchQuery(1);

        $newEntry = $result[0];
        $originalCN = $entry->getAttribute('cn')[0];

        try {
            $this->assertContains($originalCN, $newEntry->getAttribute('cn'));
            $this->assertContains('Kevin', $newEntry->getAttribute('cn'));
        } finally {
            $entryManager->rename($newEntry, 'cn='.$originalCN);
        }
    }

    public function testLdapAddRemoveAttributeValues()
    {
        $entryManager = $this->adapter->getEntryManager();

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

        $this->assertCount(2, $classConstraints);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraints[0]);
        $this->assertInstanceOf(UniqueEntity::class, $classConstraints[1]);
        $this->assertSame(['alreadyMappedUnique'], $classConstraints[0]->fields);
        $this->assertSame('unique', $classConstraints[1]->fields);

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
        $result = $this->executeSearchQuery(1);
        $newNewEntry = $result[0];

        $this->assertCount(2, $newNewEntry->getAttribute('mail'));
    }

    public function testLdapRemoveAttributeValuesError()
    {
        $entryManager = $this->adapter->getEntryManager();

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];

        $this->expectException(LdapException::class);

        $entryManager->removeAttributeValues($entry, 'mail', ['fabpot@example.org']);
    }

    public function testLdapApplyOperationsRemoveAll()
    {
        $entryManager = $this->adapter->getEntryManager();

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];

        $entryManager->applyOperations($entry->getDn(), [new UpdateOperation(\LDAP_MODIFY_BATCH_REMOVE_ALL, 'mail', null)]);

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];

        $this->assertNull($entry->getAttribute('mail'));

        $entryManager->addAttributeValues($entry, 'mail', ['fabpot@symfony.com', 'fabien@potencier.com']);
    }

    public function testLdapApplyOperationsRemoveAllWithArrayError()
    {
        $entryManager = $this->adapter->getEntryManager();

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];

        $this->expectException(UpdateOperationException::class);

        $entryManager->applyOperations($entry->getDn(), [new UpdateOperation(\LDAP_MODIFY_BATCH_REMOVE_ALL, 'mail', [])]);
    }

    public function testLdapApplyOperationsWithWrongConstantError()
    {
        $entryManager = $this->adapter->getEntryManager();

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];

        $this->expectException(UpdateOperationException::class);

        $entryManager->applyOperations($entry->getDn(), [new UpdateOperation(512, 'mail', [])]);
    }

    public function testApplyOperationsAddRemoveAttributeValues()
    {
        $entryManager = $this->adapter->getEntryManager();

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];

        $entryManager->applyOperations($entry->getDn(), [
            new UpdateOperation(\LDAP_MODIFY_BATCH_ADD, 'mail', ['fabpot@example.org', 'fabpot2@example.org']),
            new UpdateOperation(\LDAP_MODIFY_BATCH_ADD, 'mail', ['fabpot3@example.org', 'fabpot4@example.org']),
        ]);

        $result = $this->executeSearchQuery(1);
        $newEntry = $result[0];

        $this->assertCount(6, $newEntry->getAttribute('mail'));

        $entryManager->applyOperations($entry->getDn(), [
            new UpdateOperation(\LDAP_MODIFY_BATCH_REMOVE, 'mail', ['fabpot@example.org', 'fabpot2@example.org']),
            new UpdateOperation(\LDAP_MODIFY_BATCH_REMOVE, 'mail', ['fabpot3@example.org', 'fabpot4@example.org']),
        ]);

        $result = $this->executeSearchQuery(1);
        $newNewEntry = $result[0];

        $this->assertCount(2, $newNewEntry->getAttribute('mail'));
    }

    public function testUpdateOperationsWithIterator()
    {
        $result = $this->executeSearchQuery(1);
        $newEntry = $result[0];

        $this->assertCount(6, $newEntry->getAttribute('mail'));

        $entryManager->applyOperations($entry->getDn(), $iteratorRemove);

        $result = $this->executeSearchQuery(1);
        $newNewEntry = $result[0];

        $this->assertCount(2, $newNewEntry->getAttribute('mail'));
    }

    public function testUpdateOperationsThrowsExceptionWhenAddedDuplicatedValue()
    {
        $duplicateIterator = new \ArrayIterator([
            new UpdateOperation(\LDAP_MODIFY_BATCH_ADD, 'mail', ['fabpot@example.org']),
            new UpdateOperation(\LDAP_MODIFY_BATCH_ADD, 'mail', ['fabpot@example.org']),
        ]);

        $entryManager = $this->adapter->getEntryManager();

        $result = $this->executeSearchQuery(1);
        $entry = $result[0];

        $this->expectException(UpdateOperationException::class);

        $entryManager->applyOperations($entry->getDn(), $duplicateIterator);
    }

    /**
     * @group functional
     */
    public function testLdapMove()
    {
        $result = $this->executeSearchQuery(1);

        $entry = $result[0];
        $this->assertStringNotContainsString('ou=Ldap', $entry->getDn());

        $entryManager = $this->adapter->getEntryManager();
        $entryManager->move($entry, 'ou=Ldap,ou=Components,dc=symfony,dc=com');

        $result = $this->executeSearchQuery(1);
        $movedEntry = $result[0];
        $this->assertStringContainsString('ou=Ldap', $movedEntry->getDn());

        // Move back entry
        $entryManager->move($movedEntry, 'dc=symfony,dc=com');
    }
}
