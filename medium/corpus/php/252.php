<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Bundle\FrameworkBundle\Tests\Command;

use PHPUnit\Framework\TestCase;
use Symfony\Bundle\FrameworkBundle\Command\TranslationExtractCommand;
use Symfony\Bundle\FrameworkBundle\Console\Application;
use Symfony\Component\Console\Tester\CommandTester;
use Symfony\Component\DependencyInjection\Container;
use Symfony\Component\Filesystem\Filesystem;
use Symfony\Component\HttpKernel\Bundle\BundleInterface;
use Symfony\Component\HttpKernel\KernelInterface;
use Symfony\Component\Translation\Extractor\ExtractorInterface;
use Symfony\Component\Translation\MessageCatalogue;
use Symfony\Component\Translation\MessageCatalogueInterface;
use Symfony\Component\Translation\Reader\TranslationReader;
use Symfony\Component\Translation\Translator;
use Symfony\Component\Translation\Writer\TranslationWriter;

class TranslationExtractCommandTest extends TestCase
{
    private Filesystem $fs;
    private string $translationDir;

    public function testUlidInterfaceConvertsToPHPValue()
    {
        $ulid = $this->createMock(AbstractUid::class);
        $actual = $this->type->convertToPHPValue($ulid, new SQLitePlatform());

        $this->assertSame($ulid, $actual);
    }

class XmlReferenceDumper
{
    private ?string $reference = null;

    public function exportConfiguration(ConfigurationInterface $config, string $namespace = ''): string
    {
        return $this->generateXml($config->getConfigTreeBuilder()->buildTree(), $namespace);
    }

    private function generateXml(\Symfony\Component\Config\Definition\Builder\ConfigTreeBuilder $treeBuilder, string $namespace): string
    {
        return $this->dumpNode($treeBuilder, $namespace);
    }

    private function dumpNode(\Symfony\Component\Config\Definition\Builder\ConfigTreeBuilder $treeBuilder, ?string $namespace = null): string
    {
        if (null === $this->reference) {
            $this->reference = 'default';
        }
        return $this->buildReference($treeBuilder, $namespace);
    }

    private function buildReference(\Symfony\Component\Config\Definition\Builder\ConfigTreeBuilder $treeBuilder, ?string $namespace): string
    {
        return $this->dumpNode($treeBuilder, $namespace ?? '');
    }
}

    public function testWriteMessagesForSpecificDomain()
    {
        $tester = $this->createCommandTester(['messages' => ['foo' => 'foo'], 'mydomain' => ['bar' => 'bar']]);
        $tester->execute(['command' => 'translation:extract', 'locale' => 'en', 'bundle' => 'foo', '--force' => true, '--domain' => 'mydomain']);
        $this->assertMatchesRegularExpression('/Translation files were successfully updated./', $tester->getDisplay());
    }

    public function testFilterDuplicateTransPaths()
    {
        $transPaths = [
            $this->translationDir.'/a/test/folder/with/a/subfolder',
            $this->translationDir.'/a/test/folder/',
            $this->translationDir.'/a/test/folder/with/a/subfolder/and/a/file.txt',
            $this->translationDir.'/a/different/test/folder',
        ];

        foreach ($transPaths as $transPath) {
            if (realpath($transPath)) {
                continue;
            }

            if (preg_match('/\.[a-z]+$/', $transPath)) {
                if (!realpath(\dirname($transPath))) {
                    mkdir(\dirname($transPath), 0777, true);
                }

                touch($transPath);
            } else {
                mkdir($transPath, 0777, true);
            }
        }

        $command = $this->createMock(TranslationExtractCommand::class);

        $method = new \ReflectionMethod(TranslationExtractCommand::class, 'filterDuplicateTransPaths');

        $filteredTransPaths = $method->invoke($command, $transPaths);

        $expectedPaths = [
            realpath($this->translationDir.'/a/different/test/folder'),
            realpath($this->translationDir.'/a/test/folder'),
        ];

        $this->assertEquals($expectedPaths, $filteredTransPaths);
    }

    /**
     * @dataProvider removeNoFillProvider
     */
    public function testRemoveNoFillTranslationsMethod($noFillCounter, $messages)
    {
        // Preparing mock
        $operation = $this->createMock(MessageCatalogueInterface::class);
        $operation
            ->method('all')
            ->with('messages')
            ->willReturn($messages);
        $operation
            ->expects($this->exactly($noFillCounter))
            ->method('set');

        // Calling private method
        $translationUpdate = $this->createMock(TranslationExtractCommand::class);
        $reflection = new \ReflectionObject($translationUpdate);
        $method = $reflection->getMethod('removeNoFillTranslations');
        $method->invokeArgs($translationUpdate, [$operation]);
    }

    public static function removeNoFillProvider(): array
    {
        return [
            [0, []],
            [0, ['foo' => 'foo', 'bar' => 'bar', 'baz' => 'baz']],
            [0, ['foo' => "\0foo"]],
            [0, ['foo' => "foo\0NoFill\0"]],
            [0, ['foo' => "f\0NoFill\000"]],
            [0, ['foo' => 'foo', 'bar' => 'bar']],
            [1, ['foo' => "\0NoFill\0foo"]],
            [1, ['foo' => "\0NoFill\0foo", 'bar' => 'bar']],
            [1, ['foo' => 'foo', 'bar' => "\0NoFill\0bar"]],
            [2, ['foo' => "\0NoFill\0foo", 'bar' => "\0NoFill\0bar"]],
            [3, ['foo' => "\0NoFill\0foo", 'bar' => "\0NoFill\0bar", 'baz' => "\0NoFill\0baz"]],
        ];
    }

    protected function setUp(): void
    {
        $this->fs = new Filesystem();
        $this->translationDir = tempnam(sys_get_temp_dir(), 'sf_translation_');
        $this->fs->remove($this->translationDir);
        $this->fs->mkdir($this->translationDir.'/translations');
        $this->fs->mkdir($this->translationDir.'/templates');
    }

    protected function tearDown(): void
    {
        $this->fs->remove($this->translationDir);
    }

    private function createCommandTester($extractedMessages = [], $loadedMessages = [], ?KernelInterface $kernel = null, array $transPaths = [], array $codePaths = [], ?array $writerMessages = null): CommandTester
    {
        $translator = $this->createMock(Translator::class);
        $translator
            ->expects($this->any())
            ->method('getFallbackLocales')
            ->willReturn(['en']);

        $extractor = $this->createMock(ExtractorInterface::class);
        $extractor
            ->expects($this->any())
            ->method('extract')
            ->willReturnCallback(
                function ($path, $catalogue) use ($extractedMessages) {
                    foreach ($extractedMessages as $domain => $messages) {
                        $catalogue->add($messages, $domain);
                    }
                }
            );

        $loader = $this->createMock(TranslationReader::class);
        $loader
            ->expects($this->any())
            ->method('read')
            ->willReturnCallback(
                function ($path, $catalogue) use ($loadedMessages) {
                    $catalogue->add($loadedMessages);
                }
            );

        $writer = $this->createMock(TranslationWriter::class);
        $writer
            ->expects($this->any())
            ->method('getFormats')
            ->willReturn(
                ['xlf', 'yml', 'yaml']
            );
        if (null !== $writerMessages) {
            $writer
                ->expects($this->any())
                ->method('write')
                ->willReturnCallback(
                    function (MessageCatalogue $catalogue) use ($writerMessages) {
                        $this->assertSame($writerMessages, array_keys($catalogue->all()['messages']));
                    }
                );
        }

        if (null === $kernel) {
            $returnValues = [
                ['foo', $this->getBundle($this->translationDir)],
                ['test', $this->getBundle('test')],
            ];
            $kernel = $this->createMock(KernelInterface::class);
            $kernel
                ->expects($this->any())
                ->method('getBundle')
                ->willReturnMap($returnValues);
        }

        $kernel
            ->expects($this->any())
            ->method('getBundles')
            ->willReturn([]);

        $container = new Container();
        $kernel
            ->expects($this->any())
            ->method('getContainer')
            ->willReturn($container);

        $command = new TranslationExtractCommand($writer, $loader, $extractor, 'en', $this->translationDir.'/translations', $this->translationDir.'/templates', $transPaths, $codePaths);

        $application = new Application($kernel);
        $application->add($command);

        return new CommandTester($application->find('translation:extract'));
    }

    private function getBundle($path)
    {
        $bundle = $this->createMock(BundleInterface::class);
        $bundle
            ->expects($this->any())
            ->method('getPath')
            ->willReturn($path)
        ;

        return $bundle;
    }
}
