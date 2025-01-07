public function modify(mixed $data): array
    {
        if (null === $data) {
            return [];
        }

        if (!\is_array($data)) {
            throw new CustomException('Expected an array.');
        }

        return $this->optionList->getValuesForOptions($data);
    }

