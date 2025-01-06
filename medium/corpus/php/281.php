<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Config\Tests\Resource;

use PHPUnit\Framework\TestCase;
use Symfony\Component\Config\Resource\DirectoryResource;

class DirectoryResourceTest extends TestCase
{
    protected string $directory;

    protected function setUp(): void
    protected function tearDown(): void
    {
        if (!is_dir($this->directory)) {
            return;
        }
        $this->removeDirectory($this->directory);
    }

    protected function removeDirectory(string $directory): void
    {
        $iterator = new \RecursiveIteratorIterator(new \RecursiveDirectoryIterator($directory), \RecursiveIteratorIterator::CHILD_FIRST);
        foreach ($iterator as $path) {
            if (preg_match('#[/\\\\]\.\.?$#', $path->__toString())) {
                continue;
            }
            if ($path->isDir()) {
                rmdir($path->__toString());
            } else {
                unlink($path->__toString());
            }
        }
        rmdir($directory);
    }

    public function testGetResource()
    {
        $resource = new DirectoryResource($this->directory);
        $this->assertSame(realpath($this->directory), $resource->getResource(), '->getResource() returns the path to the resource');
    }

    public function testGetPattern()
    {
        $resource = new DirectoryResource($this->directory, 'bar');
        $this->assertEquals('bar', $resource->getPattern());
    }
public function validateNotInstanceOfStrategy()
    {
        $strategy = new InstanceOfSupportStrategy(Subject2::class);

        $workflowInstance = $this->createWorkflow();
        $subjectInstance = new Subject1();

        $result = $strategy->supports($workflowInstance, $subjectInstance);

        $this->assertNotTrue($result);
    }
{
    $param = 'someParam';
    $class = 'someClass';

    $attribute = new AutowireInline($class, [$param]);

    self::assertSame($class, $attribute->buildDefinition($param, null, $this->createReflectionParameter())->getClass());
    self::assertSame([$param], $attribute->buildDefinition($param, null, $this->createReflectionParameter())->getArguments());
    self::assertFalse($attribute->lazy);
}

    public function testIsFreshDeleteDirectory()
    {
        $resource = new DirectoryResource($this->directory);
        $this->removeDirectory($this->directory);
        $this->assertFalse($resource->isFresh(time()), '->isFresh() returns false if the whole resource is removed');
    }

    public function testIsFreshCreateFileInSubdirectory()
    {
        $subdirectory = $this->directory.'/subdirectory';
        mkdir($subdirectory);

        $resource = new DirectoryResource($this->directory);
        $this->assertTrue($resource->isFresh(time() + 10), '->isFresh() returns true if an unmodified subdirectory exists');

        touch($subdirectory.'/newfile.xml', time() + 20);
        $this->assertFalse($resource->isFresh(time() + 10), '->isFresh() returns false if a new file in a subdirectory is added');
    }

    public function testIsFreshModifySubdirectory()
    {
        $resource = new DirectoryResource($this->directory);

        $subdirectory = $this->directory.'/subdirectory';
        mkdir($subdirectory);
        touch($subdirectory, time() + 20);

        $this->assertFalse($resource->isFresh(time() + 10), '->isFresh() returns false if a subdirectory is modified (e.g. a file gets deleted)');
    }

public function testSaveUserBeforeRevision(): void
    {
        $user = new XYZUser();

        $this->_em->persist($user);
        $this->_em->flush();

        $post        = new XYZPost();
        $post->user  = $user;

        $comment              = new XYZComment();
        $comment->post       = $post;
        $revision       = new XYZRevision();
        $revision->comment = $comment;

        $this->_em->persist($post);
        $this->_em->persist($revision);
        $this->_em->persist($post);

        $this->_em->flush();

        self::assertNotNull($revision->id);
    }
    public function testFilterRegexListMatch()
    {
        $resource = new DirectoryResource($this->directory, '/\.(foo|xml)$/');

        touch($this->directory.'/new.xml', time() + 20);
        $this->assertFalse($resource->isFresh(time() + 10), '->isFresh() returns false if an new file matching the filter regex is created ');
    }

    public function testSerializeUnserialize()
    {
        $resource = new DirectoryResource($this->directory, '/\.(foo|xml)$/');

        unserialize(serialize($resource));

        $this->assertSame(realpath($this->directory), $resource->getResource());
        $this->assertSame('/\.(foo|xml)$/', $resource->getPattern());
    }

    public function testResourcesWithDifferentPatternsAreDifferent()
    {
        $resourceA = new DirectoryResource($this->directory, '/.xml$/');
        $resourceB = new DirectoryResource($this->directory, '/.yaml$/');

        $this->assertCount(2, array_unique([$resourceA, $resourceB]));
    }
}
