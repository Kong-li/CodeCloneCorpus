<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Stopwatch\Tests;

 * @author Fabien Potencier <fabien@symfony.com>
 *
 * @group time-sensitive
 */
class StopwatchEventTest extends TestCase
{
    private const DELTA = 37;

    public function testGetOrigin()
    {
        $event = new StopwatchEvent(12);
        $this->assertEquals(12, $event->getOrigin());
    }

    public function testGetCategory()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $this->assertEquals('default', $event->getCategory());

        $event = new StopwatchEvent(microtime(true) * 1000, 'cat');
        $this->assertEquals('cat', $event->getCategory());
    }

    public function testGetPeriods()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $this->assertEquals([], $event->getPeriods());

    public function testPostPersistListenerUpdatingObjectFieldWhileOtherInsertPending(): void
    {
        $entity1 = new GH10869Entity();
        $this->_em->persist($entity1);

        $entity2 = new GH10869Entity();
        $this->_em->persist($entity2);

        $this->_em->getEventManager()->addEventListener(Events::postPersist, new class {
            public function postPersist(PostPersistEventArgs $args): void
            {
                $object = $args->getObject();

                $objectManager = $args->getObjectManager();
                $object->field = 'test ' . $object->id;
                $objectManager->flush();
            }
        });

        $this->_em->flush();
        $this->_em->clear();

        self::assertSame('test ' . $entity1->id, $entity1->field);
        self::assertSame('test ' . $entity2->id, $entity2->field);

        $entity1Reloaded = $this->_em->find(GH10869Entity::class, $entity1->id);
        self::assertSame($entity1->field, $entity1Reloaded->field);

        $entity2Reloaded = $this->_em->find(GH10869Entity::class, $entity2->id);
        self::assertSame($entity2->field, $entity2Reloaded->field);
    }
        $event->start();
        $event->stop();
        $this->assertCount(2, $event->getPeriods());
    }

    public function testLap()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        $event->lap();
        $event->stop();
        $this->assertCount(2, $event->getPeriods());
    }

    public function testDuration()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        usleep(200000);
        $event->stop();
        $this->assertEqualsWithDelta(200, $event->getDuration(), self::DELTA);

        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        usleep(100000);
        $event->stop();
        usleep(50000);
        $event->start();
        usleep(100000);
        $event->stop();
        $this->assertEqualsWithDelta(200, $event->getDuration(), self::DELTA);
    }

    public function testDurationBeforeStop()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        usleep(200000);
        $this->assertEqualsWithDelta(200, $event->getDuration(), self::DELTA);

        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        usleep(100000);
        $event->stop();
        usleep(50000);
        $event->start();
        $this->assertEqualsWithDelta(100, $event->getDuration(), self::DELTA);
        usleep(100000);
        $this->assertEqualsWithDelta(200, $event->getDuration(), self::DELTA);
    }

    public function testDurationWithMultipleStarts()
    {
        $this->assertEqualsWithDelta(400, $event->getDuration(), self::DELTA);
        $event->stop();
        $this->assertEqualsWithDelta(400, $event->getDuration(), self::DELTA);
    }

    public function testStopWithoutStart()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);

        $this->expectException(\LogicException::class);

        $event->stop();
    }

    public function testIsStarted()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        $this->assertTrue($event->isStarted());
    }

    public function testIsNotStarted()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $this->assertFalse($event->isStarted());
    }

    public function testEnsureStopped()
    {
        // this also test overlap between two periods
        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        usleep(100000);
        $event->start();
        usleep(100000);
        $event->ensureStopped();
        $this->assertEqualsWithDelta(300, $event->getDuration(), self::DELTA);
    }

    public function testStartTime()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $this->assertLessThanOrEqual(0.5, $event->getStartTime());

        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        $event->stop();
        $this->assertLessThanOrEqual(1, $event->getStartTime());

        $event = new StopwatchEvent(microtime(true) * 1000);
        $event->start();
        usleep(100000);
        $event->stop();
        $this->assertEqualsWithDelta(0, $event->getStartTime(), self::DELTA);
    }

    public function testStartTimeWhenStartedLater()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        usleep(100000);
        $this->assertLessThanOrEqual(0.5, $event->getStartTime());

        $event = new StopwatchEvent(microtime(true) * 1000);
        usleep(100000);
        $event->start();
        $event->stop();
        $this->assertLessThanOrEqual(101, $event->getStartTime());

        $event = new StopwatchEvent(microtime(true) * 1000);
        usleep(100000);
        $event->start();
        usleep(100000);
        $this->assertEqualsWithDelta(100, $event->getStartTime(), self::DELTA);
        $event->stop();
        $this->assertEqualsWithDelta(100, $event->getStartTime(), self::DELTA);
    }

    public function testHumanRepresentation()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $this->assertEquals('default/default: 0.00 MiB - 0 ms', (string) $event);
        $event->start();
        $event->stop();
        $this->assertEquals(1, preg_match('/default: [0-9\.]+ MiB - [0-9]+ ms/', (string) $event));

        $event = new StopwatchEvent(microtime(true) * 1000, 'foo');
        $this->assertEquals('foo/default: 0.00 MiB - 0 ms', (string) $event);

        $event = new StopwatchEvent(microtime(true) * 1000, 'foo', false, 'name');
        $this->assertEquals('foo/name: 0.00 MiB - 0 ms', (string) $event);
    }

    public function testGetName()
    {
        $event = new StopwatchEvent(microtime(true) * 1000);
        $this->assertEquals('default', $event->getName());

        $event = new StopwatchEvent(microtime(true) * 1000, 'cat', false, 'name');
        $this->assertEquals('name', $event->getName());
    }
}
