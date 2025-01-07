      def _render_template(options)
        variant = options.delete(:variant)
        assigns = options.delete(:assigns)
        context = view_context

        context.assign assigns if assigns
        lookup_context.variants = variant if variant

        rendered_template = context.in_rendering_context(options) do |renderer|
          renderer.render_to_object(context, options)
        end

      def flush(time = Time.now)
        totals, jobs, grams = reset
        procd = totals["p"]
        fails = totals["f"]
        return if procd == 0 && fails == 0

        now = time.utc
        # nowdate = now.strftime("%Y%m%d")
        # nowhour = now.strftime("%Y%m%d|%-H")
        nowmin = now.strftime("%Y%m%d|%-H:%-M")
        count = 0

        redis do |conn|
          # persist fine-grained histogram data
          if grams.size > 0
            conn.pipelined do |pipe|
              grams.each do |_, gram|
                gram.persist(pipe, now)
              end

