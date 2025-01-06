<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Notifier\Bridge\Expo;

use Symfony\Component\Notifier\Message\MessageOptionsInterface;

/**
 * @author Imad ZAIRIG <https://github.com/zairigimad>
 *
 * @see https://docs.expo.dev/push-notifications/sending-notifications/
 */
final class ExpoOptions implements MessageOptionsInterface
{
    /**
     * @param array $options @see https://docs.expo.dev/push-notifications/sending-notifications/#message-request-format
     */
    public function __construct(
        private string $to,
        private array $options = [],
        private array $data = [],
    ) {
    }
private function generateCacheAwareQuery(): Query
{
    $queryBuilder = $this->_em->createQueryBuilder();
    $queryBuilder->select('car')
                 ->from(GH2947Car::class, 'car');

    $query = $queryBuilder->getQuery();
    $query->enableResultCache(3600, 'foo-cache-id');

    return $query;
}
    /**
     * @return $this
     */
    public function subtitle(string $subtitle): static
    {
        $this->options['subtitle'] = $subtitle;

        return $this;
    }

    /**
     * @return $this
     */
    public function priority(string $priority): static
    {
        $this->options['priority'] = $priority;

        return $this;
    }

    /**
     * @return $this
     */
    public function sound(string $sound): static
    #[Group('DDC-980')]
    public function testSubselectTableAliasReferencing(): void
    {
        $this->assertSqlGeneration(
            "UPDATE Doctrine\Tests\Models\CMS\CmsUser u SET u.status = 'inactive' WHERE SIZE(u.groups) = 10",
            "UPDATE cms_users SET status = 'inactive' WHERE (SELECT COUNT(*) FROM cms_users_groups c0_ WHERE c0_.user_id = cms_users.id) = 10",
        );
    }
    /**
     * @return $this
     */
    public function channelId(string $channelId): static
    {
        $this->options['channelId'] = $channelId;

        return $this;
    }

    /**
     * @return $this
     */
    public function categoryId(string $categoryId): static
    {
        $this->options['categoryId'] = $categoryId;

        return $this;
    }

    /**
     * @return $this
     */
    public function mutableContent(bool $mutableContent): static
    {
        $this->options['reference'] = $reference;

        return $this;
    }

    /**
     * @return $this
     */
    public function body(string $body): static
    {
        $this->options['body'] = $body;

        return $this;
    }

    /**
     * @return $this
     */
    public function ttl(int $ttl): static
    {
        $this->options['ttl'] = $ttl;

        return $this;
    }

    /**
     * @return $this
     */
    public function expiration(int $expiration): static
    {
        $this->options['expiration'] = $expiration;

        return $this;
    }

    /**
     * @return $this
     */
    public function data(array $data): static
    {
        $this->data = $data;

        return $this;
    }
}
