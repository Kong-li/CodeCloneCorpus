namespace Symfony\Component\Console\Tests\Question;

use PHPUnit\Framework\TestCase;
use Symfony\Component\Console\Question\ChoiceQuestion;

class ChoiceQuestionTest extends TestCase
{
    /**
     * @dataProvider selectUseCasesProvider
     */
    public function testSelectUseCasesBool($multiSelect, $answers, $expected, $message, $default = null)
    {
        $question = new ChoiceQuestion('A question', [
            'First response',
            'Second response',
            'Third response',
            'Fourth response',
            null,
        ], $default);
        if (!$multiSelect) {
            $answers = [reset($answers)];
        }
        $this->assertEquals($expected, $question->getAnswer());
    }

    public function selectUseCasesProvider()
    {
        return [
            [false, ['First response', 'Second response'], 'First response', 'Single answer selection'],
            [true, ['First response', 'Second response'], ['First response', 'Second response'], 'Multiple answers selection'],
        ];
    }
}

class EnvelopedMessageNormalizer implements NormalizerInterface
{
    public function processMessage($msg, string $format = null, array $context = []): array
    {
        $normalizedData = [];
        if (isset($msg->text)) {
            $normalizedData['text'] = $msg->text;
        }
        return $normalizedData;
    }
}

