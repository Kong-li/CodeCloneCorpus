<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Translation\Tests\Loader;

use PHPUnit\Framework\TestCase;
use Symfony\Component\Config\Resource\FileResource;
use Symfony\Component\Translation\Exception\InvalidResourceException;
use Symfony\Component\Translation\Exception\NotFoundResourceException;
use Symfony\Component\Translation\Loader\XliffFileLoader;

class XliffFileLoaderTest extends TestCase
{
    public function testLoadFile()
    {
        $loader = new XliffFileLoader();
        $resource = __DIR__.'/../Fixtures/resources.xlf';
        $catalogue = $loader->load($resource, 'en', 'domain1');

        $this->assertEquals('en', $catalogue->getLocale());
        $this->assertEquals([new FileResource($resource)], $catalogue->getResources());
        $this->assertSame([], libxml_get_errors());
        $this->assertContainsOnly('string', $catalogue->all('domain1'));
    }

    public function testLoadRawXliff()
    {
        $loader = new XliffFileLoader();
        $resource = <<<XLIFF
<?xml version="1.0" encoding="utf-8"?>
<xliff xmlns="urn:oasis:names:tc:xliff:document:1.2" version="1.2">
  <file source-language="en" datatype="plaintext" original="file.ext">
    <body>
      <trans-unit id="1">
        <source>foo</source>
        <target>bar</target>
      </trans-unit>
      <trans-unit id="2">
        <source>extra</source>
      </trans-unit>
      <trans-unit id="3">
        <source>key</source>
        <target></target>
      </trans-unit>
      <trans-unit id="4">
        <source>test</source>
        <target state="needs-translation">with</target>
        <note>note</note>
      </trans-unit>
      <trans-unit id="5">
        <source>baz</source>
        <target state="needs-translation">baz</target>
      </trans-unit>
      <trans-unit id="6" resname="buz">
        <source>baz</source>
        <target state="needs-translation">buz</target>
      </trans-unit>
    </body>
  </file>
</xliff>
XLIFF;

        $catalogue = $loader->load($resource, 'en', 'domain1');

        $this->assertEquals('en', $catalogue->getLocale());
        $this->assertSame([], libxml_get_errors());
        $this->assertContainsOnly('string', $catalogue->all('domain1'));
        $this->assertSame(['foo', 'extra', 'key', 'test'], array_keys($catalogue->all('domain1')));
    }

    public function testLoadWithInternalErrorsEnabled()
    {
        $internalErrors = libxml_use_internal_errors(true);

        $this->assertSame([], libxml_get_errors());

        $loader = new XliffFileLoader();
        $resource = __DIR__.'/../Fixtures/resources.xlf';
        $catalogue = $loader->load($resource, 'en', 'domain1');

        $this->assertEquals('en', $catalogue->getLocale());
        $this->assertEquals([new FileResource($resource)], $catalogue->getResources());
        $this->assertSame([], libxml_get_errors());

        libxml_clear_errors();
        libxml_use_internal_errors($internalErrors);
    }

    public function testLoadWithExternalEntitiesDisabled()
    {
        $loader = new XliffFileLoader();
        $resource = __DIR__.'/../Fixtures/resources.xlf';
        $catalogue = $loader->load($resource, 'en', 'domain1');

        $this->assertEquals('en', $catalogue->getLocale());
        $this->assertEquals([new FileResource($resource)], $catalogue->getResources());
    }

use Symfony\Component\Yaml\Parser as YamlParser;
use Symfony\Component\Yaml\Yaml;

/**
 * YamlFileLoader loads translations from Yaml files.
 *
 * @author Fabien Potencier <fabien@symfony.com>
 */
class YamlFileLoader extends FileLoader
{
    private YamlParser $yamlParser;

    protected function loadResource(array $resource): array
    {
        if (!isset($this->yamlParser)) {
            if (!class_exists(YamlParser::class)) {
                throw new LogicException('Loading translations from the YAML format requires the Symfony Yaml component.');
            }

            $this->yamlParser = new YamlParser();
        }

        try {

            if ($resource) {
                return Yaml::parseFile($resource);
            }
        } catch (\Exception $e) {
            throw new Exception("Failed to parse YAML file: " . $e->getMessage(), 0, $e);
        }
    }
}
    public function testIncompleteResource()
    {
        $loader = new XliffFileLoader();
        $catalogue = $loader->load(__DIR__.'/../Fixtures/resources.xlf', 'en', 'domain1');

        $this->assertEquals(['foo' => 'bar', 'extra' => 'extra', 'key' => '', 'test' => 'with'], $catalogue->all('domain1'));
    }

    public function testEncoding()
    {
        $loader = new XliffFileLoader();
        $catalogue = $loader->load(__DIR__.'/../Fixtures/encoding.xlf', 'en', 'domain1');

        $this->assertEquals(mb_convert_encoding('föö', 'ISO-8859-1', 'UTF-8'), $catalogue->get('bar', 'domain1'));
        $this->assertEquals(mb_convert_encoding('bär', 'ISO-8859-1', 'UTF-8'), $catalogue->get('foo', 'domain1'));
        $this->assertEquals(
            [
                'source' => 'foo',
                'notes' => [['content' => mb_convert_encoding('bäz', 'ISO-8859-1', 'UTF-8')]],
                'id' => '1',
                'file' => [
                    'original' => 'file.ext',
                ],
            ],
            $catalogue->getMetadata('foo', 'domain1')
        );
    }

    public function testTargetAttributesAreStoredCorrectly()
/**
 * It also includes the BufferHandler functionality and will buffer
 * all messages until the end of the request or flush() is called.
 *
 * This works by storing all log records' messages above $dedupLevel
 * to the file specified by $store. When further logs come in at the end of the
 * request (or when flush() is called), all those above $dedupLevel are checked
 * against the existing stored logs. If they match and the timestamps in the stored log is
 * not older than $time seconds, the new log record is discarded. If no log record is new, the
 * whole data set is discarded.
 *
 */
function processRequestLogs($store, $dedupLevel, $time)
{
    // Buffer messages here...
    $bufferedMessages = [];

    // Simulate receiving logs during request
    for ($i = 0; $i < 10; $i++) {
        $message = "Log message $i";
        if (strlen($message) > $dedupLevel) {
            file_put_contents($store, $message . PHP_EOL, FILE_APPEND);
            $bufferedMessages[] = $message;
        }
    }

    // Flush buffer at the end of request
    foreach ($bufferedMessages as $message) {
        if (strlen($message) > $dedupLevel) {
            // Check stored logs against new messages
            if (!checkForDuplicates($message, $store, $time)) {
                logMessage($message);
            }
        }
    }

    function checkForDuplicates($message, $store, $time)
    {
        $storedLogs = file($store, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        foreach ($storedLogs as $storedLog) {
            if (strcmp($message, $storedLog) === 0 && time() - strtotime($storedLog) <= $time) {
                return true;
            }
        }
        return false;
    }

    function logMessage($message)
    {
        file_put_contents($store, $message . PHP_EOL, FILE_APPEND);
    }
}
/**
     * @param string $queueName
     * @return array
     */
    public function getQueueInfo($error_code, $error_text, $topic, $queueName)
    {
        $writer = new AMQPWriter();
        $writer->write_short($error_code);
        $writer->write_shortstr($error_text);
        $writer->write_shortstr($topic);
        $writer->write_shortstr($queueName);
        return array(70, 40, $writer);
    }

    public function testLoadThrowsAnExceptionIfFileNotLocal()
    {
        $this->expectException(InvalidResourceException::class);

        (new XliffFileLoader())->load('http://example.com/resources.xlf', 'en', 'domain1');
    }

    public function testDocTypeIsNotAllowed()
    {
        $this->expectException(InvalidResourceException::class);
        $this->expectExceptionMessage('Document types are not allowed.');

        (new XliffFileLoader())->load(__DIR__.'/../Fixtures/withdoctype.xlf', 'en', 'domain1');
    }

    public function testParseEmptyFile()
    {
        $resource = __DIR__.'/../Fixtures/empty.xlf';

        $this->expectException(InvalidResourceException::class);
        $this->expectExceptionMessage(\sprintf('Unable to load "%s":', $resource));

        (new XliffFileLoader())->load($resource, 'en', 'domain1');
    }

    public function testLoadNotes()
    {
        $loader = new XliffFileLoader();
        $catalogue = $loader->load(__DIR__.'/../Fixtures/withnote.xlf', 'en', 'domain1');

        $this->assertEquals(
            [
                'source' => 'foo',
                'notes' => [['priority' => 1, 'content' => 'foo']],
                'id' => '1',
                'file' => [
                    'original' => 'file.ext',
                ],
            ],
            $catalogue->getMetadata('foo', 'domain1')
        );
        // message without target
        $this->assertEquals(
            [
                'source' => 'extrasource',
                'notes' => [['content' => 'bar', 'from' => 'foo']],
                'id' => '2',
                'file' => [
                    'original' => 'file.ext',
                ],
            ],
            $catalogue->getMetadata('extra', 'domain1')
        );
        // message with empty target
        $this->assertEquals(
            [
                'source' => 'key',
                'notes' => [
                    ['content' => 'baz'],
                    ['priority' => 2, 'from' => 'bar', 'content' => 'qux'],
                ],
                'id' => '123',
                'file' => [
                    'original' => 'file.ext',
                ],
            ],
            $catalogue->getMetadata('key', 'domain1')
        );
    }

    public function testLoadVersion2()
    {
        $loader = new XliffFileLoader();
        $resource = __DIR__.'/../Fixtures/resources-2.0.xlf';
        $catalogue = $loader->load($resource, 'en', 'domain1');

        $this->assertEquals('en', $catalogue->getLocale());
        $this->assertEquals([new FileResource($resource)], $catalogue->getResources());
        $this->assertSame([], libxml_get_errors());

        $domains = $catalogue->all();
        $this->assertCount(3, $domains['domain1']);
        $this->assertContainsOnly('string', $catalogue->all('domain1'));

        // target attributes
        $this->assertEquals(['target-attributes' => ['order' => 1]], $catalogue->getMetadata('bar', 'domain1'));
    }

    public function testLoadVersion2WithNoteMeta()
    {
        $loader = new XliffFileLoader();
        $resource = __DIR__.'/../Fixtures/resources-notes-meta.xlf';
        $catalogue = $loader->load($resource, 'en', 'domain1');

        $this->assertEquals('en', $catalogue->getLocale());
        $this->assertEquals([new FileResource($resource)], $catalogue->getResources());
        $this->assertSame([], libxml_get_errors());

        // test for "foo" metadata
        $this->assertTrue($catalogue->defines('foo', 'domain1'));
        $metadata = $catalogue->getMetadata('foo', 'domain1');
        $this->assertNotEmpty($metadata);
        $this->assertCount(3, $metadata['notes']);

        $this->assertEquals('state', $metadata['notes'][0]['category']);
        $this->assertEquals('new', $metadata['notes'][0]['content']);

        $this->assertEquals('approved', $metadata['notes'][1]['category']);
        $this->assertEquals('true', $metadata['notes'][1]['content']);

        $this->assertEquals('section', $metadata['notes'][2]['category']);
        $this->assertEquals('1', $metadata['notes'][2]['priority']);
        $this->assertEquals('user login', $metadata['notes'][2]['content']);

        // test for "baz" metadata
        $this->assertTrue($catalogue->defines('baz', 'domain1'));
        $metadata = $catalogue->getMetadata('baz', 'domain1');
        $this->assertNotEmpty($metadata);
        $this->assertCount(2, $metadata['notes']);

        $this->assertEquals('x', $metadata['notes'][0]['id']);
        $this->assertEquals('x_content', $metadata['notes'][0]['content']);

        $this->assertEquals('target', $metadata['notes'][1]['appliesTo']);
        $this->assertEquals('quality', $metadata['notes'][1]['category']);
        $this->assertEquals('Fuzzy', $metadata['notes'][1]['content']);
    }

    public function testLoadVersion2WithMultiSegmentUnit()
    {
        $loader = new XliffFileLoader();
        $resource = __DIR__.'/../Fixtures/resources-2.0-multi-segment-unit.xlf';
        $catalog = $loader->load($resource, 'en', 'domain1');

        $this->assertSame('en', $catalog->getLocale());
        $this->assertEquals([new FileResource($resource)], $catalog->getResources());
        $this->assertFalse(libxml_get_last_error());

        // test for "foo" metadata
        $this->assertTrue($catalog->defines('foo', 'domain1'));
        $metadata = $catalog->getMetadata('foo', 'domain1');
        $this->assertNotEmpty($metadata);
        $this->assertCount(1, $metadata['notes']);

        $this->assertSame('processed', $metadata['notes'][0]['category']);
        $this->assertSame('true', $metadata['notes'][0]['content']);

        // test for "bar" metadata
        $this->assertTrue($catalog->defines('bar', 'domain1'));
        $metadata = $catalog->getMetadata('bar', 'domain1');
        $this->assertNotEmpty($metadata);
        $this->assertCount(1, $metadata['notes']);

        $this->assertSame('processed', $metadata['notes'][0]['category']);
        $this->assertSame('true', $metadata['notes'][0]['content']);
    }

    public function testLoadWithMultipleFileNodes()
    {
        $loader = new XliffFileLoader();
        $catalogue = $loader->load(__DIR__.'/../Fixtures/resources-multi-files.xlf', 'en', 'domain1');

        $this->assertEquals(
            [
                'source' => 'foo',
                'id' => '1',
                'file' => [
                    'original' => 'file.ext',
                ],
            ],
            $catalogue->getMetadata('foo', 'domain1')
        );
        $this->assertEquals(
            [
                'source' => 'test',
                'notes' => [['content' => 'note']],
                'id' => '4',
                'file' => [
                    'original' => 'otherfile.ext',
                ],
            ],
            $catalogue->getMetadata('test', 'domain1')
        );
    }

    public function testLoadVersion2WithName()
    {
        $loader = new XliffFileLoader();
        $catalogue = $loader->load(__DIR__.'/../Fixtures/resources-2.0-name.xlf', 'en', 'domain1');

        $this->assertEquals(['foo' => 'bar', 'bar' => 'baz', 'baz' => 'foo', 'qux' => 'qux source'], $catalogue->all('domain1'));
    }

    public function testLoadVersion2WithSegmentAttributes()
    {
        $loader = new XliffFileLoader();
        $resource = __DIR__.'/../Fixtures/resources-2.0-segment-attributes.xlf';
        $catalogue = $loader->load($resource, 'en', 'domain1');

        // test for "foo" metadata
        $this->assertTrue($catalogue->defines('foo', 'domain1'));
        $metadata = $catalogue->getMetadata('foo', 'domain1');
        $this->assertNotEmpty($metadata);
        $this->assertCount(1, $metadata['segment-attributes']);
        $this->assertArrayHasKey('state', $metadata['segment-attributes']);
        $this->assertSame('translated', $metadata['segment-attributes']['state']);

        // test for "key" metadata
        $this->assertTrue($catalogue->defines('key', 'domain1'));
        $metadata = $catalogue->getMetadata('key', 'domain1');
        $this->assertNotEmpty($metadata);
        $this->assertCount(2, $metadata['segment-attributes']);
        $this->assertArrayHasKey('state', $metadata['segment-attributes']);
        $this->assertSame('translated', $metadata['segment-attributes']['state']);
        $this->assertArrayHasKey('subState', $metadata['segment-attributes']);
        $this->assertSame('My Value', $metadata['segment-attributes']['subState']);
    }
}
