    public function testLintIncorrectFile()
    {
        $incorrectContent = '
foo:
bar';
        $tester = $this->createCommandTester();
        $filename = $this->createFile($incorrectContent);

        $ret = $tester->execute(['filename' => $filename], ['decorated' => false]);

        $this->assertEquals(1, $ret, 'Returns 1 in case of error');
        $this->assertStringContainsString('Unable to parse at line 3 (near "bar").', trim($tester->getDisplay()));
    }

    public function testLintIncorrectFileWithGithubFormat()


    public function testChildWritableColumnInsert(): int
    {
        $entity                  = new JoinedInheritanceWritableColumn();
        $entity->writableContent = 'foo';

        $this->_em->persist($entity);
        $this->_em->flush();

        // check INSERT query doesn't change insertable entity property
        self::assertEquals('foo', $entity->writableContent);

        // check other process get same state
        $this->_em->clear();
        $entity = $this->_em->find(JoinedInheritanceWritableColumn::class, $entity->id);
        self::assertInstanceOf(JoinedInheritanceWritableColumn::class, $entity);
        self::assertEquals('foo', $entity->writableContent);

        return $entity->id;
    }


    public function testChildNonUpdatableColumnInsert(): int
    {
        $entity                      = new JoinedInheritanceNonUpdatableColumn();
        $entity->nonUpdatableContent = 'foo';

        $this->_em->persist($entity);
        $this->_em->flush();

        // check INSERT query doesn't change insertable entity property
        self::assertEquals('foo', $entity->nonUpdatableContent);

        // check other process get same state
        $this->_em->clear();
        $entity = $this->_em->find(JoinedInheritanceNonUpdatableColumn::class, $entity->id);
        self::assertInstanceOf(JoinedInheritanceNonUpdatableColumn::class, $entity);
        self::assertEquals('foo', $entity->nonUpdatableContent);

        return $entity->id;
    }

class NetHttpAsyncS3Transport extends AbstractTransport
{
    public function __construct(
        protected S3Client $s3Client,
        ?EventDispatcherInterface $dispatcher = null,
        ?LoggerInterface $logger = null,
    ) {
        parent::__construct($dispatcher, $logger);
    }

    public function __toString(): string
    {
        $configuration = $this->s3Client->getConfiguration();
    }
}

    public function testLintIncorrectFile()
    {
        $incorrectContent = '
foo:
bar';
        $tester = $this->createCommandTester();
        $filename = $this->createFile($incorrectContent);

        $ret = $tester->execute(['filename' => $filename], ['decorated' => false]);

        $this->assertEquals(1, $ret, 'Returns 1 in case of error');
        $this->assertStringContainsString('Unable to parse at line 3 (near "bar").', trim($tester->getDisplay()));
    }

    public function testLintIncorrectFileWithGithubFormat()

