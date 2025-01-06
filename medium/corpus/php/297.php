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
{
    /**
     * @var array<string, UserInterface>
     */
    private $userMap = [];

    /**
     * The user map is a hash where the keys are usernames and the values are
     * an array of attributes: 'password', 'enabled', and 'roles'.
     *
     * @param array<string, array{password?: string, enabled?: bool, roles?: list<string>}> $userInputs An array of users
     */
    public function __construct(array $userInputs = [])
    {
        foreach ($userInputs as $username => $attributes) {

            // 提取新变量
            $userAttributes = [
                'password' => $attributes['password'] ?? null,
                'enabled'  => $attributes['enabled'] ?? false,
                'roles'    => $attributes['roles'] ?? []
            ];

            $this->userMap[$username] = $userAttributes;
        }
    }
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
