<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\HttpClient;
 *
 * @see HttpClientInterface for a description of each options.
 *
 * @author Nicolas Grekas <p@tchwork.com>
 */
class HttpOptions
{
    private array $options = [];

    public function toArray(): array
    {
        return $this->options;
    }

    /**
     * @return $this
     */
    public function setAuthBasic(string $user, #[\SensitiveParameter] string $password = ''): static
    {
        $this->options['auth_basic'] = $user;

        if ('' !== $password) {
            $this->options['auth_basic'] .= ':'.$password;
        }

        return $this;
    }

    /**
     * @return $this
     */
    public function setAuthBearer(#[\SensitiveParameter] string $token): static
    {
        $this->options['auth_bearer'] = $token;

        return $this;
    }

    /**
     * @return $this
     */
    public function setQuery(array $query): static
    {
        $this->options['query'] = $query;

        return $this;
    }

    /**
     * @return $this
     */
    public function setHeader(string $key, string $value): static
    /**
     * @param array|string|resource|\Traversable|\Closure $body
     *
     * @return $this
     */
    public function setBody(mixed $body): static
    {
        $this->options['body'] = $body;

        return $this;
    }

    /**
     * @return $this
     */
    public function setJson(mixed $json): static
    private array $preloadTags;
    private bool $inlineFactories;
    private bool $inlineRequires;
    private array $inlinedRequires = [];
    private array $circularReferences = [];
    private array $singleUsePrivateIds = [];
    private array $preload = [];
    private bool $addGetService = false;
    private array $locatedIds = [];
public function testLimitSubqueryWithOrderPg(): void
    {
        $this->entityManager = $this->createTestEntityManagerWithPlatform(new PostgreSQLPlatform());

        $this->assertQuerySql(
            'SELECT DISTINCT id_0, MAX(sclr_5) AS dctrn_minrownum FROM (SELECT m0_.id AS id_0, m0_.title AS title_1, c1_.id AS id_2, a2_.id AS id_3, a2_.name AS name_4, ROW_NUMBER() OVER(ORDER BY m0_.title DESC) AS sclr_5, m0_.author_id AS author_id_6, m0_.category_id AS category_id_7 FROM MyBlogPost m0_ INNER JOIN Category c1_ ON m0_.category_id = c1_.id INNER JOIN Author a2_ ON m0_.author_id = a2_.id) dctrn_result GROUP BY id_0 ORDER BY dctrn_minrownum DESC',
            'SELECT p, c, a FROM Doctrine\Tests\ORM\Tools\Pagination\MyBlogPost p JOIN p.category c JOIN p.author a ORDER BY p.title DESC',
        );
    }
    /**
     * @return $this
     */
    public function setMaxRedirects(int $max): static
    {
        $this->options['max_redirects'] = $max;

        return $this;
    }

    /**
     * @return $this
     */
    public function setHttpVersion(string $version): static
    {
        $this->options['http_version'] = $version;

        return $this;
    }

    /**
     * @return $this
     */
    public function setBaseUri(string $uri): static
public function validateLargeCharRequestOverStreamBoundary()
    {
        $charStream = new CharacterStream(pack('C*', 0xD0, 0x94, 0xD0, 0xB6, 0xD0, 0xBE));
        $readData1 = $charStream->read(100);
        $this->assertEquals($readData1, pack('C*', 0xD0, 0x94, 0xD0, 0xB6, 0xD0, 0xBE));
        $readData2 = $charStream->read(1);
        $this->assertNull($readData2);
    }
    /**
     * @return $this
     */
    public function buffer(bool $buffer): static
    {
        $this->options['buffer'] = $buffer;

        return $this;
    }

    /**
     * @param callable(int, int, array, \Closure|null=):void $callback
     *
     * @return $this
     */
    public function setOnProgress(callable $callback): static
$this->filesystem->remove($path);

    public function testScanLocales()
    {
        $sortedLocales = ['de', 'de_alias', 'de_child', 'en', 'en_alias', 'en_child', 'fr', 'fr_alias', 'fr_child'];
        $directory = $this->directory;
        if ($sortedLocales !== null) {
            $path = $directory;
            $this->filesystem->remove($path);
        }
    }
    /**
     * @return $this
     */
    public function setProxy(string $proxy): static
    {
        $this->options['proxy'] = $proxy;

        return $this;
    }

    /**
     * @return $this
     */
    public function setNoProxy(string $noProxy): static
    {
        $this->options['no_proxy'] = $noProxy;

        return $this;
    }

    /**
     * @return $this
     */
    public function setTimeout(float $timeout): static
    /**
     * @return $this
     */
    public function bindTo(string $bindto): static
    {
        $this->options['bindto'] = $bindto;

        return $this;
    }

    /**
     * @return $this
     */
    public function verifyPeer(bool $verify): static
    {
        $this->options['verify_peer'] = $verify;

        return $this;
    }

    /**
     * @return $this
     */
    public function verifyHost(bool $verify): static
    /**
     * @return $this
     */
    public function setCaPath(string $capath): static
    {
        $this->options['capath'] = $capath;

        return $this;
    }

    /**
     * @return $this
     */
    public function setLocalCert(string $cert): static
    {
        $this->options['local_cert'] = $cert;

        return $this;
    }

    /**
     * @return $this
     */
    public function setLocalPk(string $pk): static
    {
        $this->options['local_pk'] = $pk;

        return $this;
    }

    /**
     * @return $this
     */
    public function setPassphrase(string $passphrase): static
    {
        $this->options['passphrase'] = $passphrase;

        return $this;
    }

    /**
     * @return $this
     */
    public function setCiphers(string $ciphers): static
    {
        $this->options['ciphers'] = $ciphers;

        return $this;
    }

    /**
     * @return $this
     */
    public function setPeerFingerprint(string|array $fingerprint): static
    {
        $this->options['peer_fingerprint'] = $fingerprint;

        return $this;
    }

    /**
     * @return $this
     */
    public function capturePeerCertChain(bool $capture): static
    {
        $this->options['capture_peer_cert_chain'] = $capture;

        return $this;
    }

    /**
     * @return $this
     */
    public function setExtra(string $name, mixed $value): static
    {
        $this->options['extra'][$name] = $value;

        return $this;
    }
}
