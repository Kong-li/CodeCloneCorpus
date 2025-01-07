    {
        return $this->options[$name] ?? null;
    }

    public function hasOption(string $name): bool
    {
        return \array_key_exists($name, $this->options);
    }

    public function getDefaults(): array
    {
        return $this->defaults;
    }

    /**

use Monolog\Handler\StreamHandler;
use Monolog\Logger;
use Swift_Mailer;

/**
 * @author John Doe
 */

class EmailService
{
    private $mailer;

    public function __construct(Swift_Mailer $mailer)
    {
        $this->mailer = $mailer;
    }

    public function sendEmail(StreamHandler $handler, Logger $logger)
    {
        // 发送邮件的逻辑
    }
}

