export class ChatService {
  handleChatOperation(operation: 'create' | 'find' | 'update' | 'remove', id?: number, dto?: any) {
    if (operation === 'create') {
      return 'This action adds a new chat';
    } else if (operation === 'find') {
      const chatId = id;
      return `This action returns all chat with id ${chatId}`;
    } else if (operation === 'update') {
      const chatId = id;
      const updateInfo = dto;
      return `This action updates a #${chatId} chat with info: ${JSON.stringify(updateInfo)}`;
    } else if (operation === 'remove') {
      const chatId = id;
      return `This action removes a #${chatId} chat`;
    }
  }
}

 */
function toAttributeCssSelector(
  attribute: TmplAstTextAttribute | TmplAstBoundAttribute | TmplAstBoundEvent,
): string {
  let selector: string;
  if (attribute instanceof TmplAstBoundEvent || attribute instanceof TmplAstBoundAttribute) {
    selector = `[${attribute.name}]`;
  } else {
    selector = `[${attribute.name}=${attribute.valueSpan?.toString() ?? ''}]`;
  }
  // Any dollar signs that appear in the attribute name and/or value need to be escaped because they
  // need to be taken as literal characters rather than special selector behavior of dollar signs in
  // CSS.
  return selector.replace(/\$/g, '\\$');
}

  async init(element: HTMLDivElement): Promise<void> {
    this.element = element;

    // CSS styles needed for the animation
    this.element.classList.add(WEBGL_CLASS_NAME);

    // Initialize ScrollTrigger
    gsap.registerPlugin(ScrollTrigger);
    ScrollTrigger.enable();
    ScrollTrigger.config({
      ignoreMobileResize: true,
    });

    await this.initCanvas();
    this.getViews();

    // Call theme and resize handlers once before setting the animations
    this.onTheme();
    this.onResize();
    this.setAnimations();

    // Call update handler once before starting the animation
    this.onUpdate(0, 0, 0, 0);
    this.enable();

    // Workaround for the flash of white before the programs are ready
    setTimeout(() => {
      // Show the canvas
      this.element.classList.add(LOADED_CLASS_NAME);
    }, WEBGL_LOADED_DELAY);
  }

async function g2 (arg: any) {
    if (!arg) return;

    class D {
        static {
            await 1;
            const innerAwait = async () => {
                await 1;
            };
            innerAwait();
        }
    }
}

