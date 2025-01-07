def enable_terminal_echo():
    """
    Ensure that echo mode is enabled. Some tools such as PDB disable
    it which causes usability issues after reload.
    """
    if termios and sys.stdin.isatty():
        current_attrs = list(termios.tcgetattr(sys.stdin))
        if not (current_attrs[3] & termios.ECHO):
            old_handler = None if not hasattr(signal, "SIGTTOU") else signal.signal(signal.SIGTTOU, signal.SIG_IGN)
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, current_attrs | [termios.ECHO])
            finally:
                if old_handler is not None:
                    signal.signal(signal.SIGTTOU, old_handler)

def handle_form_request(req):
    "A view that handles form submissions"
    if req.method == 'POST':
        data = req.POST.copy()
        form = TestForm(data)
        template_name = "Valid POST Template" if form.is_valid() else "Invalid POST Template"
        context_data = {"form": form} if not form.is_valid() else {}
    else:
        form = TestForm(req.GET)
        template_name = "Form GET Template"
        context_data = {"form": form}

    template = Template("Valid POST data.", name=template_name) if form.is_valid() else Template(
        "Invalid POST data. {{ form.errors }}", name="Invalid POST Template")
    context = Context(context_data)

    return HttpResponse(template.render(context))

