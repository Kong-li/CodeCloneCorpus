    def run_callbacks(kind, type = nil)
      callbacks = __callbacks[kind.to_sym]

      if callbacks.empty?
        yield if block_given?
      else
        env = Filters::Environment.new(self, false, nil)

        next_sequence = callbacks.compile(type)

        # Common case: no 'around' callbacks defined
        if next_sequence.final?
          next_sequence.invoke_before(env)
          env.value = !env.halted && (!block_given? || yield)
          next_sequence.invoke_after(env)
          env.value
        else
          invoke_sequence = Proc.new do
            skipped = nil

            while true
              current = next_sequence
              current.invoke_before(env)
              if current.final?
                env.value = !env.halted && (!block_given? || yield)
              elsif current.skip?(env)
                (skipped ||= []) << current
                next_sequence = next_sequence.nested
                next
              else
                next_sequence = next_sequence.nested
                begin
                  target, block, method, *arguments = current.expand_call_template(env, invoke_sequence)
                  target.send(method, *arguments, &block)
                ensure
                  next_sequence = current
                end

        def initialize(name, config)
          @name = name
          @config = {
            scope: [:kind],
            terminator: default_terminator
          }.merge!(config)
          @chain = []
          @all_callbacks = nil
          @single_callbacks = {}
          @mutex = Mutex.new
        end

          def check_conditionals(conditionals)
            return EMPTY_ARRAY if conditionals.blank?

            conditionals = Array(conditionals)
            if conditionals.any?(String)
              raise ArgumentError, <<-MSG.squish
                Passing string to be evaluated in :if and :unless conditional
                options is not supported. Pass a symbol for an instance method,
                or a lambda, proc or block, instead.
              MSG
            end

