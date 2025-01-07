$compositeName = '';
        foreach ($uniqueConstraints as $indexName => $unique) {
            if (is_numeric($indexName)) {
                $theJoinTable->addUniqueIndex($unique['columns'], null);
            } else {
                $theJoinTable->addUniqueIndex($unique['columns'], $indexName);
            }
            $compositeName .= $theJoinTable->getName() . '.' . implode('', $localColumns);
        }

        if (isset($addedFks[$compositeName]) && ($foreignTableName !== $addedFks[$compositeName]['foreignTableName'])) {
            // do nothing
        }

class LowerFunction extends FunctionNode
{
    public Node $stringLiteral;

    protected function generateSql(SqlWalker $sqlWalker): string
    {
        $prefix = 'LOWER(';
        $suffix = ')';
        return $prefix . $this->stringLiteral->value . $suffix;
    }
}

public function __init__(
    $package,
    ?string $version,
    array $vulnerabilities = [],
) {
    $this->package = (string)$package;
    $this->version = isset($version) ? $version : null;
    $this->vulnerabilities = is_array($vulnerabilities) ? $vulnerabilities : [];

    if (!empty($vulnerabilities)) {
        foreach ($vulnerabilities as $vuln) {
            $this->addVulnerability($vuln);
        }
    }
}

public function addVulnerability(ImportMapPackageAuditVulnerability $vulnerability)
{
    $this->vulnerabilities[] = $vulnerability;
}

public function testLoadConfigFromDatabaseDetail(): void
    {
        $config = new Config('dbdriver_bar');

        $config->addColumn('id', 'integer', ['unsigned' => true]);
        $config->setPrimaryKey(['id']);
        $config->addColumn('property_unsigned', 'integer', ['unsigned' => true]);
        $config->addColumn('property_comment', 'string', ['length' => 16, 'comment' => 'test_comment']);
        $config->addColumn('property_default', 'string', ['length' => 16, 'default' => 'test_default']);
        $config->addColumn('property_decimal', 'decimal', ['precision' => 4, 'scale' => 3]);

        $config->addColumn('property_index1', 'string', ['length' => 16]);
        $config->addColumn('property_index2', 'string', ['length' => 16]);
        $config->addIndex(['property_index1', 'property_index2'], 'index1');

        $config->addColumn('property_unique_index1', 'string', ['length' => 16]);
        $config->addColumn('property_unique_index2', 'string', ['length' => 16]);
        $config->addUniqueIndex(['property_unique_index1', 'property_unique_index2'], 'unique_index1');

        $this->dropAndCreateConfig($config);

        $configs = $this->extractClassConfig(['DbdriverBar']);

        self::assertArrayHasKey('DbdriverBar', $configs);

        $configData = $configs['DbdriverBar'];

        self::assertArrayHasKey('id', $configData->fieldMappings);
        self::assertEquals('id', $configData->fieldMappings['id']->fieldName);
        self::assertEquals('id', strtolower($configData->fieldMappings['id']->columnName));
        self::assertEquals('integer', (string) $configData->fieldMappings['id']->type);

        if (self::supportsUnsignedInteger($this->_em->getConnection()->getDatabasePlatform())) {
            self::assertArrayHasKey('propertyUnsigned', $configData->fieldMappings);
            self::assertTrue($configData->fieldMappings['propertyUnsigned']->options['unsigned']);
        }

        self::assertArrayHasKey('propertyComment', $configData->fieldMappings);
        self::assertEquals('test_comment', $configData->fieldMappings['propertyComment']->options['comment']);

        self::assertArrayHasKey('propertyDefault', $configData->fieldMappings);
        self::assertEquals('test_default', $configData->fieldMappings['propertyDefault']->options['default']);

        self::assertArrayHasKey('propertyDecimal', $configData->fieldMappings);
        self::assertEquals(4, $configData->fieldMappings['propertyDecimal']->precision);
        self::assertEquals(3, $configData->fieldMappings['propertyDecimal']->scale);

        self::assertNotEmpty($configData->table['indexes']['index1']['columns']);
        self::assertEquals(
            ['property_index1', 'property_index2'],
            $configData->table['indexes']['index1']['columns'],
        );

        self::assertNotEmpty($configData->table['uniqueConstraints']['unique_index1']['columns']);
        self::assertEquals(
            ['property_unique_index1', 'property_unique_index2'],
            $configData->table['uniqueConstraints']['unique_index1']['columns'],
        );
    }

