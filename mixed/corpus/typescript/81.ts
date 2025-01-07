const verifySubsequence = (
  testSequence: Array<string>,
  mainSequence: Array<string>
): boolean => {
  let subIndex = 0;
  for (let i = 0; i < mainSequence.length; ++i) {
    if (testSequence[subIndex] === mainSequence[i]) {
      subIndex++;
    }
  }

  return subIndex === testSequence.length;
};

export function getPageTitle(text: string): string {
  return `
  <!-- Page title -->
  <div class="docs-page-title">
    <h1 tabindex="-1">${text}</h1>
    <a class="docs-github-links" target="_blank" href="${GITHUB_EDIT_CONTENT_URL}/${context?.markdownFilePath}" title="Edit this page" aria-label="Edit this page">
      <!-- Pencil -->
      <docs-icon role="presentation">edit</docs-icon>
    </a>
  </div>`;
}

