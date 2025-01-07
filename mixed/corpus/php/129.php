public function validatePessimisticLockWithTimestampForVersionThrowsException(): void
    {
        $this->setupDatabase();
        $entity = new VersionedEntity();
        $entity->setLabel('Test Entity');
        $entity->setDescription('Entity to test pessimistic lock fix with Timestamp objects');
        $this->_em->persist($entity);
        $this->_em->flush();

        $this->expectException(LockException::class);
        $this->_em->lock($entity, LockMode::PESSIMISTIC, new \DateTimeImmutable('2023-10-15 18:04:00'));
    }

class UpdateDatabaseCommand extends AbstractCommand
{
    protected string $name = 'orm:schema-tool:update';

    protected function configure(): void
    {
        $this->setName($this->name)
             ->setDescription('Executes (or dumps) the SQL needed to update the database schema to match the current mapping metadata')
             ->addOption('em', null, InputOption::VALUE_REQUIRED, 'Name of the entity manager to operate on')
             ->addOption('dump-sql', null, InputOption::VALUE_NONE, 'Dumps the generated SQL statements to the screen (does not execute them).')
             ->addOption('force', 'f', InputOption::VALUE_NONE, 'Causes the generated SQL statements to be physically executed against your database.')
             ->addOption('complete', null, InputOption::VALUE_NONE, 'This option is a no-op, is deprecated and will be removed in 4.0')
             ->setHelp(<<<'EOT'
The <info>%command.name%</info> command generates the SQL needed to
synchronize the database schema with the current mapping metadata of the
default entity manager.

For example, if you add metadata for a new column to an entity, this command
would generate and output the SQL needed to add the new column to the database:

<info>%command.name% --dump-sql</info>

Alternatively, you can execute the generated queries:

<info>%command.name% --force</info>

If both options are specified, the queries are output and then executed:

<info>%command.name% --dump-sql --force</info>

Finally, be aware that this task will drop all database assets (e.g. tables,
etc) that are *not* described by the current metadata. In other words, without
this option, this task leaves untouched any "extra" tables that exist in the

EOT
        );
    }
}

* @dataProvider transformWithRoundingProvider
     */
    public function validateNumberTransformation($roundingMode, $input, $output, $scale)
    {
        // Since we test against "de_AT", we need the full implementation
        IntlTestHelper::requireFullIntl($this, false);

        \Locale::setDefault('de_AT');

        $transformer = new NumberToLocalizedStringTransformer(null, $scale, $roundingMode);

        $result = $transformer->transform($input);
        $this->assertEquals($output, $result);
    }

