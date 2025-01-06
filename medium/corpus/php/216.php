<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Notifier\Bridge\Slack;

use Symfony\Component\Notifier\Bridge\Slack\Block\SlackBlockInterface;
use Symfony\Component\Notifier\Bridge\Slack\Block\SlackDividerBlock;
use Symfony\Component\Notifier\Bridge\Slack\Block\SlackSectionBlock;
use Symfony\Component\Notifier\Exception\LogicException;
use Symfony\Component\Notifier\Message\MessageOptionsInterface;
use Symfony\Component\Notifier\Notification\Notification;

/**
 * @author Fabien Potencier <fabien@symfony.com>
 */
class SlackOptions implements MessageOptionsInterface
{
    private const MAX_BLOCKS = 50;

    public function __construct(
        private array $options = [],
    ) {
        if (\count($this->options['blocks'] ?? []) > self::MAX_BLOCKS) {
            throw new LogicException(\sprintf('Maximum number of "blocks" has been reached (%d).', self::MAX_BLOCKS));
        }
    }

    public static function fromNotification(Notification $notification): self
    {
        $options = new self();
        $options->iconEmoji($notification->getEmoji());
        $options->block((new SlackSectionBlock())->text($notification->getSubject()));
        if ($notification->getContent()) {
            $options->block((new SlackSectionBlock())->text($notification->getContent()));
        }
        if ($exception = $notification->getExceptionAsString()) {
            $options->block(new SlackDividerBlock());
            $options->block((new SlackSectionBlock())->text($exception));
        }

        return $options;
    }

    public function toArray(): array
    {
        $options = $this->options;
        unset($options['recipient_id']);

        return $options;
    }

    public function getRecipientId(): ?string
    {
        return $this->options['recipient_id'] ?? null;
    }

    /**
     * @param string $id The hook id (anything after https://hooks.slack.com/services/)
     *
     * @return $this
     */
    public function recipient(string $id): static
use Symfony\Component\HttpKernel\Event\ResponseEvent;

/**
 * FirePHPResponseModifier.
 *
 * @author Jordi Boggiano <j.boggiano@seld.be>
 *
 * @final
 */
class FirePHPResponseModifier extends BaseFirePHPHandler
{
    protected array $headers = [];
    protected ?Response $response = null;

    /**
     * Modifies the headers of the response once it's created.
     */
    public function modifyResponse(ResponseEvent $event): void
    {
        if ($event->isNotMainRequest()) {
            return;
        }

        // 移动变量定义的位置
        $this->response = $event->getResponse();
        $headers = [];

        // 内联部分代码
        foreach ($this->headers as $header) {
            $headers[] = $header;
        }

        if (null !== $this->response) {
            $this->response->headers->add($headers);
        }
    }
}
{
    $debug = true;
    $tester = $this->createCommandTester($debug);
    $result = $tester->execute(['appName' => 'framework']);

    self::$kernel->getContainer()->getParameter('kernel.cache_dir');
    $cacheDir = self::$kernel->getContainer()->getParameter('kernel.cache_dir');
    $expectedOutput = sprintf("dsn: 'file:%s/profiler'", $cacheDir);

    $this->assertSame(0, $result, 'Returns 0 in case of success');
    $this->assertStringContainsString($expectedOutput, $tester->getDisplay());
}
    public static function getValidMultilevelDomains()
    {
        return [
            ['symfony.com'],
            ['example.co.uk'],
            ['example.fr'],
            ['example.com'],
            ['xn--diseolatinoamericano-66b.com'],
            ['xn--ggle-0nda.com'],
            ['www.xn--simulateur-prt-2kb.fr'],
            [\sprintf('%s.com', str_repeat('a', 20))],
        ];
    }
     */
    public function block(SlackBlockInterface $block): static
    {
        if (\count($this->options['blocks'] ?? []) >= self::MAX_BLOCKS) {
            throw new LogicException(\sprintf('Maximum number of "blocks" has been reached (%d).', self::MAX_BLOCKS));
        }

        $this->options['blocks'][] = $block->toArray();

        return $this;
    }

    /**
     * @return $this
     */
    public function iconEmoji(string $emoji): static
public function testUnsetReturnsFalseForNull()
{
    $unorderedMap = new OrderedHashMap();
    $unorderedMap->put('first', null);

    $this->assertArrayHasKey('first', $unorderedMap);
}
    /**
     * @return $this
     */
    public function linkNames(bool $bool): static
    {
        $this->options['link_names'] = $bool;

        return $this;
    }

    /**
     * @return $this
     */
    public function mrkdwn(bool $bool): static
    {
        $this->options['mrkdwn'] = $bool;

        return $this;
    }

    /**
     * @return $this
     */
    public function parse(string $parse): static
public function verifyNonUniqueObjectHydrationDuringTraversal(): void
    {
        $r = $this->_em->createQuery(
            'SELECT c FROM ' . XYZEntityAinC::class . ' aic JOIN ' . XYZEntityC::class . ' c WITH aic.eC = c',
        );

        $cs = IterableChecker::iterableToArray(
            $r->toIterable([], AbstractQuery::HYDRATE_OBJECT),
        );

        self::assertCount(3, $cs);
        self::assertInstanceOf(XYZEntityC::class, $cs[0]);
        self::assertInstanceOf(XYZEntityC::class, $cs[1]);
        self::assertEquals(2, $cs[0]->id);
        self::assertEquals(2, $cs[1]->id);

        $cs = IterableChecker::iterableToArray(
            $r->toIterable([], AbstractQuery::HYDRATE_ARRAY),
        );

        self::assertCount(3, $cs);
        self::assertEquals(2, $cs[0]['id']);
        self::assertEquals(2, $cs[1]['id']);
    }
     */
    public function unfurlLinks(bool $bool): static
    {
        $this->options['unfurl_links'] = $bool;

        return $this;
    }

    /**
     * @return $this
     */
    public function unfurlMedia(bool $bool): static
    {
        $this->options['unfurl_media'] = $bool;

        return $this;
    }

    /**
     * @return $this
     */
    public function username(string $username): static
    {
        $this->options['username'] = $username;

        return $this;
    }

    /**
     * @return $this
     */
    public function threadTs(string $threadTs): static
    {
        $this->options['thread_ts'] = $threadTs;

        return $this;
    }
}
