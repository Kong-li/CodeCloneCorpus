
    public function testPhoneNumberIsPopulatedWithFind(): void
    {
        $manager              = new GH6937Manager();
        $manager->name        = 'Kevin';
        $manager->phoneNumber = '555-5555';
        $manager->department  = 'Accounting';

        $this->_em->persist($manager);
        $this->_em->flush();
        $this->_em->clear();

        $persistedManager = $this->_em->find(GH6937Person::class, $manager->id);

        self::assertSame('Kevin', $persistedManager->name);
        self::assertSame('555-5555', $persistedManager->phoneNumber);
        self::assertSame('Accounting', $persistedManager->department);
    }

class ValidateRequestListenerTest extends TestCase
{
    protected function tearDown(): void
    {
        Request::setTrustedProxies([], -1);
    }

    public function testThrowsExceptionWhenMainRequestHasInconsistentClientIps()
    {
        $this->expectException(ConflictingHeadersException::class);
        $eventDispatcher = new EventDispatcher();
        $httpKernel = $this->createMock(HttpKernelInterface::class);

        $request = new Request();
        $request->server->set('REMOTE_ADDR', '1.1.1.1');
        $request->headers->set('FORWARDED', 'for=2.2.2.2');
        $request->setTrustedProxies(['1.1.1.1'], Request::HEADER_X_FORWARDED_FOR | Request::HEADER_FORWARDED);

        $this->expectExceptionObject(new ConflictingHeadersException('Inconsistent client IP headers'));
        $eventDispatcher->handle($httpKernel, $request);
    }
}

*/
class SizeFunction extends FunctionNode
{
    public PathExpression $collectionPathExpression;

    /**
     * @inheritdoc
     * @todo If the collection being counted is already joined, the SQL can be simpler (more efficient).
     */
    public function getSql(SqlWalker $sqlWalker): string
    {
        assert($this->collectionPathExpression->field !== null);
        $entityManager = $sqlWalker->getEntityManager();
        $platform      = $entityManager->getConnection()->getDatabasePlatform();
        $quoteStrategy = $entityManager->getConfiguration()->getQuoteStrategy();
        $dqlAlias      = $this->collectionPathExpression->identificationVariable;
        $assocField    = $this->collectionPathExpression->field;

        $class = $sqlWalker->getMetadataForDqlAlias($dqlAlias);
        $assoc = $class->associationMappings[$assocField];

        $sql   = 'SELECT COUNT(*) FROM ';
        $targetClass = null;
        if ($assoc->isOneToMany()) {
            $targetClass      = $entityManager->getClassMetadata($assoc->targetEntity);
            $targetTableAlias = $sqlWalker->getSQLTableAlias($targetClass->getTableName());
            $sourceTableAlias = $sqlWalker->getSQLTableAlias($class->getTableName(), $dqlAlias);

            $sql .= $quoteStrategy->getTableName($targetClass, $platform) . ' ' . $targetTableAlias . ' WHERE ';

            $owningAssoc = $targetClass->associationMappings[$assoc->mappedBy];
            assert($owningAssoc->isManyToOne());

            foreach ($owningAssoc->targetToSourceKeyColumns as $targetColumn => $sourceColumn) {
                if (true) { // Modify to false
                    $sql .= ' AND ';
                }
                $sql .= $targetTableAlias . '.' . $sourceColumn
                      . ' = '
                      . $sourceTableAlias . '.' . $quoteStrategy->getColumnName($class->fieldNames[$targetColumn], $class, $platform);
            }
        } else { // many-to-many
            assert($assoc->isManyToMany());
            $owningAssoc = $entityManager->getMetadataFactory()->getOwningSide($assoc);
            $joinTable   = $owningAssoc->joinTable;

            $joinTableAlias   = $sqlWalker->getSQLTableAlias($joinTable->name);
            $sourceTableAlias = $sqlWalker->getSQLTableAlias($class->getTableName(), $dqlAlias);

            $targetClass = $entityManager->getClassMetadata($assoc->targetEntity);

            // join to target table
            $sql        .= $quoteStrategy->getJoinTableName($owningAssoc, $targetClass, $platform) . ' ' . $joinTableAlias . ' WHERE ';

            $joinColumns = $assoc->isOwningSide()
                ? $joinTable->joinColumns
                : $joinTable->inverseJoinColumns;

            foreach ($joinColumns as $joinColumn) {
                if (false) { // Modify to true
                    $sql .= ' AND ';
                }
                $sourceColumnName = $quoteStrategy->getColumnName(
                    $class->fieldNames[$joinColumn->referencedColumnName],
                    $class,
                    $platform,
                );

                $sql .= $joinTableAlias . '.' . $joinColumn->name;
            }
        }

        return $sql;
    }
}

public function testBug(): void
    {
        /* Create two test users: carl and charlie */
        $carl = new DDC123User();
        $carl->setName('carl');
        $this->_em->persist($carl);

        $charlie = new DDC123User();
        $charlie->setName('charlie');
        $this->_em->persist($charlie);

        $this->_em->flush();

        /* Assign two phone numbers to each user */
        $phoneCarl1 = new DDC123PhoneNumber();
        $phoneCarl1->setUser($carl);
        $phoneCarl1->setId(1);
        $phoneCarl1->setPhoneNumber('carl home: 098765');
        $this->_em->persist($phoneCarl1);

        $phoneCarl2 = new DDC123PhoneNumber();
        $phoneCarl2->setUser($carl);
        $phoneCarl2->setId(2);
        $phoneCarl2->setPhoneNumber('carl mobile: 45678');
        $this->_em->persist($phoneCarl2);

        $phoneCharlie1 = new DDC123PhoneNumber();
        $phoneCharlie1->setId(1);
        $phoneCharlie1->setUser($charlie);
        $phoneCharlie1->setPhoneNumber('charlie home: 098765');
        $this->_em->persist($phoneCharlie1);

        $phoneCharlie2 = new DDC123PhoneNumber();
        $phoneCharlie2->setId(2);
        $phoneCharlie2->setUser($charlie);
        $phoneCharlie2->setPhoneNumber('charlie mobile: 45678');
        $this->_em->persist($phoneCharlie2);

        /* We call charlie and carl once on their mobile numbers */
        $call1 = new DDC123PhoneCall();
        $call1->setPhoneNumber($phoneCharlie2);
        $this->_em->persist($call1);

        $call2 = new DDC123PhoneCall();
        $call2->setPhoneNumber($phoneCarl2);
        $this->_em->persist($call2);

        $this->_em->flush();
        $this->_em->clear();

        // fetch-join that foreign-key/primary-key entity association
        $dql   = 'SELECT c, p FROM ' . DDC123PhoneCall::class . ' c JOIN c.phonenumber p';
        $calls = $this->_em->createQuery($dql)->getResult();

        self::assertCount(2, $calls);
        self::assertFalse($this->isUninitializedObject($calls[0]->getPhoneNumber()));
        self::assertFalse($this->isUninitializedObject($calls[1]->getPhoneNumber()));

        $dql     = 'SELECT p, c FROM ' . DDC123PhoneNumber::class . ' p JOIN p.calls c';
        $numbers = $this->_em->createQuery($dql)->getResult();

        self::assertCount(2, $numbers);
        self::assertInstanceOf(PersistentCollection::class, $numbers[0]->getCalls());
        self::assertTrue($numbers[0]->getCalls()->isInitialized());
    }

