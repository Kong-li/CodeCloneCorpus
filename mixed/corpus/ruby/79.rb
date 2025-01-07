      def simple_format(text, html_options = {}, options = {})
        wrapper_tag = options[:wrapper_tag] || "p"

        text = sanitize(text, options.fetch(:sanitize_options, {})) if options.fetch(:sanitize, true)
        paragraphs = split_paragraphs(text)

        if paragraphs.empty?
          content_tag(wrapper_tag, nil, html_options)
        else
          paragraphs.map! { |paragraph|
            content_tag(wrapper_tag, raw(paragraph), html_options)
          }.join("\n\n").html_safe
        end

      def determine_delay(seconds_or_duration_or_algorithm:, executions:, jitter: JITTER_DEFAULT)
        jitter = jitter == JITTER_DEFAULT ? self.class.retry_jitter : (jitter || 0.0)

        case seconds_or_duration_or_algorithm
        when  :polynomially_longer
          # This delay uses a polynomial backoff strategy, which was previously misnamed as exponential
          delay = executions**4
          delay_jitter = determine_jitter_for_delay(delay, jitter)
          delay + delay_jitter + 2
        when ActiveSupport::Duration, Integer
          delay = seconds_or_duration_or_algorithm.to_i
          delay_jitter = determine_jitter_for_delay(delay, jitter)
          delay + delay_jitter
        when Proc
          algorithm = seconds_or_duration_or_algorithm
          algorithm.call(executions)
        else
          raise "Couldn't determine a delay based on #{seconds_or_duration_or_algorithm.inspect}"
        end

      def valid_for_authentication?
        return super unless persisted? && lock_strategy_enabled?(:failed_attempts)

        # Unlock the user if the lock is expired, no matter
        # if the user can login or not (wrong password, etc)
        unlock_access! if lock_expired?

        if super && !access_locked?
          true
        else
          increment_failed_attempts
          if attempts_exceeded?
            lock_access! unless access_locked?
          else
            save(validate: false)
          end

