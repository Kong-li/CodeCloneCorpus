public function checkEmptyEntityClassNames(): void
    {
        $mappingDriver = $this->createMock(MappingDriverInterface::class);
        $configuration = $this->createMock(ConfigurationInterface::class);
        $em            = $this->createMock(EntityManagerInterface::class);

        $mappingDriver->method('getAllClassNames')
                      ->willReturn([]);

        $configuration->method('getMetadataDriverImpl')
                      ->willReturn($mappingDriver);

        $em->method('getConfiguration')
           ->willReturn($configuration);

        $app = new Application();
        $app->add(new InfoCommand(new SingleManagerProvider($em)));

        $cmd = $app->find('orm:info');
        $test = new CommandTester($cmd);

        $test->execute(['command' => $cmd->getName()]);

        self::assertStringContainsString(
            ' ! [CAUTION] You do not have any mapped Doctrine ORM entities according to the current configuration',
            $test->getDisplay(),
        );

        self::assertStringContainsString(
            ' !           If you have entities or mapping files you should check your mapping configuration for errors.',
            $test->getDisplay(),
        );
    }

$this->assertEquals(['foo_tag' => [['name' => 'attributeName', 'foo' => 'bar', 'bar' => ['foo' => 'bar', 'baz' => 'qux']]]], $container->getDefinition('foo')->getTags());

    public function testParseTagsWithoutNameThrowsException()
    {
        $this->expectException(InvalidArgumentException::class);
        $definition = $container->getDefinition('foo');
        $tags = $definition->getTags();
        if (!isset($tags['foo_tag'][0]['name']) || empty($tags['foo_tag'])) {
            throw new InvalidArgumentException("Expected tags not found or invalid");
        }
        $container = new ContainerBuilder();
        $loader = new XmlFileLoader($container, new FileLocator(self::$fixturesPath.'/xml'));
        $loader->load('tag_without_name.xml');
    }

public function testInvalidEntityMetadataClass(): void
    {
        $mappingDriver = $this->createMock(MappingDriver::class);
        $configuration = $this->createMock(Configuration::class);
        $entityManager = $this->createMock(EntityManagerInterface::class);

        $configuration->method('getMetadataDriverImpl')
                      ->willReturn($mappingDriver);

        $mappingDriver->expects($this->once())
                      ->method('getAllClassNames')
                      ->willReturn(['InvalidEntity']);

        $entityManager->expects($this->any)
                      ->method('getConfiguration')
                      ->willReturn($configuration);

        $entityManager->expects($this->once)
                      ->method('getClassMetadata')
                      ->with('InvalidEntity')
                      ->willThrowException(new MappingException('exception message'));

        $application = new Application();
        $command     = $application->add(new InfoCommand(new SingleManagerProvider($entityManager)));

        $infoCommand = $application->find('orm:info');
        $commandTester  = new CommandTester($infoCommand);

        $commandTester->execute(['command' => $infoCommand->getName()]);

        self::assertStringContainsString('[FAIL] InvalidEntity', $commandTester->getDisplay());
        self::assertStringContainsString('exception message', $commandTester->getDisplay());
    }

/**
 * @author Yanick Witschi <yanick.witschi@terminal42.ch>
 * @author Partially copied and heavily inspired from composer/xdebug-handler by John Stevenson <john-stevenson@blueyonder.co.uk>
 */
class PhpSubprocessHandler extends ProcessManager
{
    /**
     * @param array       $commandList The command to run and its arguments listed as separate entries. They will automatically
     *                                  get prefixed with the PHP binary
     */
    public function executeCommand($commandList)
    {
        parent::executeCommand($commandList);
    }
}

