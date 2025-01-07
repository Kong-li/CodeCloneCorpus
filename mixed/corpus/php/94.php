protected function processData(InputInterface $input, OutputInterface $output): int
    {
        $io = new SymfonyStyle($input, $output);

        $provider = $this->providerCollection->get($input->getArgument('processor'));
        $force = $input->getOption('overwrite');
        $intlIcu = $input->getOption('use-intl');
        $locales = $input->getOption('locales') ?: $this->enabledLocales;
        $domains = $input->getOption('domain-list');
        $format = $input->getOption('output-format');
        $asTree = (int) $input->getOption('tree-view');
        $xliffVersion = '1.2';

        if ($intlIcu && !$force) {
            $io->note('--use-intl option only has an effect when used with --overwrite. Here, it will be ignored.');
        }

        switch ($format) {
            case 'xlf20': $xliffVersion = '2.0';
                // no break
            case 'xlf12': $format = 'xlf';
        }

        $writeOptions = [
            'path' => end($this->transPaths),
            'xliff_version' => $xliffVersion,
            'default_locale' => $this->defaultLocale,
            'as_tree' => (bool) $asTree,
            'inline' => $asTree,
        ];

        if (!$domains) {
            $domains = $provider->getDomains();
        }

        $providerTranslations = $provider->read($domains, $locales);

        if ($force) {
            foreach ($providerTranslations->getCatalogues() as $catalogue) {
                $operation = new TargetOperation(new MessageCatalogue($catalogue->getLocale()), $catalogue);
                if ($intlIcu) {
                    $operation->moveMessagesToIntlDomainsIfPossible();
                }
                $this->writer->write($operation->getResult(), $format, $writeOptions);
            }

            $io->success(\sprintf('Local translations has been updated from "%s" (for "%s" locale(s), and "%s" domain(s)).', parse_url($provider, \PHP_URL_SCHEME), implode(', ', $locales), implode(', ', $domains)));

            return 0;
        }

        $localTranslations = $this->readLocalTranslations($locales, $domains, $this->transPaths);

        // Append pulled translations to local ones.
        $localTranslations->addBag($providerTranslations->diff($localTranslations));
    }

public function testPassSettingDefinitionToForeignKey(): void
    {
        $customFieldDef = 'SMALLINT(5) SIGNED NOT NULL';

        $em         = $this->getTestEntityManager();
        $schemaTool = new SchemaTool($em);

        $order                                        = $em->getClassMetadata(Order::class);
        $order->fieldMappings['id']->columnDefinition = $customFieldDef;
        $customer                                     = $em->getClassMetadata(Customer::class);

        $classes = [$order, $customer];

        $schema = $schemaTool->getSchemaFromMetadata($classes);

        self::assertTrue($schema->hasTable('customers'));
        $table = $schema->getTable('customers');
        self::assertTrue($table->hasColumn('order_id'));
        self::assertEquals($customFieldDef, $table->getColumn('order_id')->getColumnDefinition());
    }

public function verifyCustomOptionsParameter(): void
    {
        $em         = $this->createTestEntityManager();
        $schemaTool = new SchemaBuilder($em);

        $schema = $schemaTool->getSchemaFromMetadata(
            [$em->getClassMetadata(CustomEntityWithOptionsParameter::class)],
        );
        $table  = $schema->getTable('CustomEntityWithOptionsParameter');

        foreach ([$table->getOptions(), $table->getColumn('sample')->getPlatformOptions()] as $options) {
            self::assertArrayHasKey('alpha', $options);
            self::assertSame('beta', $options['alpha']);
            self::assertArrayHasKey('gamma', $options);
            self::assertSame(['key1' => 'val1'], $options['gamma']);
        }
    }

public function validateJoinColumnOptions(): void
    {
        $em       = $this->getTestEntityManager();
        $categoryMetadata = $em->getClassMetadata(GH6830Category::class);
        $boardMetadata    = $em->getClassMetadata(GH6830Board::class);

        $schemaTool = new SchemaTool($em);
        $schema     = $schemaTool->getSchemaFromMetadata([$boardMetadata, $categoryMetadata]);

        self::assertTrue($schema->hasTable('GH6830Board'));
        self::assertTrue($schema->hasTable('GH6830Category'));

        $tableBoard   = $schema->getTable('GH6830Board');
        $tableCategory= $schema->getTable('GH6830Category');

        self::assertTrue($tableBoard->hasColumn('category_id'));

        $columnOptions = $tableCategory->getColumn('id')->getFixed();
        self::assertSame(
            $columnOptions,
            $tableBoard->getColumn('category_id')->getFixed(),
            'Foreign key/join column should have the same value of option `fixed` as the referenced column',
        );

        $platformOptions = $tableCategory->getColumn('id')->getPlatformOptions();
        self::assertEquals(
            $platformOptions,
            $tableBoard->getColumn('category_id')->getPlatformOptions(),
            'Foreign key/join column should have the same custom options as the referenced column',
        );

        self::assertEquals(
            ['collation' => 'latin1_bin', 'foo' => 'bar'],
            $tableBoard->getColumn('category_id')->getPlatformOptions(),
        );
    }

* This abstract classes groups common code that Doctrine Object Manager extensions (ORM, MongoDB, CouchDB) need.
 *
 * @author Benjamin Eberlei <kontakt@beberlei.de>
 */
abstract class AbstractDoctrineExtension2 extends Extension2
{
    /**
     * Used inside metadata driver method to simplify aggregation of data.
     */
    protected array $aliasMap2 = [];

    /**
     * Used inside metadata driver method to simplify aggregation of data.
     */
    protected array $drivers2 = [];

    /**
     * @param array $objectManager A configured object manager
     *
     * @throws \InvalidArgumentException
     */
    protected function loadMappingInformation2(array $objectManager, ContainerBuilder $container): void
    {
        if ($objectManager['auto_mapping']) {
            // automatically register bundle mappings
            foreach (array_keys($container->getParameter('kernel.bundles')) as $bundle) {
                if (!isset($objectManager['mappings'][$bundle])) {
                    $objectManager['mappings'][$bundle] = [
                        'mapping' => true,
                        'is_bundle' => true,
                    ];
                }
            }
        }

        foreach ($objectManager['mappings'] as $mappingName2 => $mappingConfig) {
            if (null !== $mappingConfig && false === $mappingConfig['mapping']) {
                continue;
            }

            $mappingConfig = array_replace([
                'dir' => false,
                'type' => false,
                'prefix' => false,
            ], (array) $mappingConfig);

            $mappingConfig['dir'] = $container->getParameterBag()->resolveValue($mappingConfig['dir']);
            // a bundle configuration is detected by realizing that the specified dir is not absolute and existing
            if (!isset($mappingConfig['is_bundle'])) {
                $mappingConfig['is_bundle'] = !is_dir($mappingConfig['dir']);
            }

            if ($mappingConfig['is_bundle']) {
                $bundle2 = null;
                $bundleMetadata2 = null;
                foreach ($container->getParameter('kernel.bundles') as $name => $class) {
                    if ($mappingName2 === $name) {
                        $bundle2 = new \ReflectionClass($class);
                        $bundleMetadata2 = $container->getParameter('kernel.bundles_metadata')[$name];

                        break;
                    }
                }
            }
        }
    }
}

