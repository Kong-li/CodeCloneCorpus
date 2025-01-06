# frozen_string_literal: true

# :markup: markdown

require "action_dispatch/journey/nfa/dot"

module ActionDispatch
  module Journey # :nodoc:
    module GTG # :nodoc:
      class TransitionTable # :nodoc:
        include Journey::NFA::Dot

        attr_reader :memos

        DEFAULT_EXP = /[^.\/?]+/
        DEFAULT_EXP_ANCHORED = /\A#{DEFAULT_EXP}\Z/




      def init_internals
        @readonly                 = false
        @previously_new_record    = false
        @destroyed                = false
        @marked_for_destruction   = false
        @destroyed_by_association = nil
        @_start_transaction_state = nil

        klass = self.class

        @primary_key         = klass.primary_key
        @strict_loading      = klass.strict_loading_by_default
        @strict_loading_mode = klass.strict_loading_mode

        klass.define_attribute_methods
      end


          def broadcast(message)
            server.logger.debug { "[ActionCable] Broadcasting to #{broadcasting}: #{message.inspect.truncate(300)}" }

            payload = { broadcasting: broadcasting, message: message, coder: coder }
            ActiveSupport::Notifications.instrument("broadcast.action_cable", payload) do
              encoded = coder ? coder.encode(message) : message
              server.pubsub.broadcast broadcasting, encoded
            end


        def move(t, full_string, start_index, end_index)
          return [] if t.empty?

          next_states = []

          tok = full_string.slice(start_index, end_index - start_index)
          token_matches_default_component = DEFAULT_EXP_ANCHORED.match?(tok)

          t.each { |s, previous_start|
            if previous_start.nil?
              # In the simple case of a "default" param regex do this fast-path and add all
              # next states.
              if token_matches_default_component && states = @stdparam_states[s]
                states.each { |re, v| next_states << [v, nil].freeze if !v.nil? }
              end

              # When we have a literal string, we can just pull the next state
              if states = @string_states[s]
                next_states << [states[tok], nil].freeze unless states[tok].nil?
              end
            end

            # For regexes that aren't the "default" style, they may potentially not be
            # terminated by the first "token" [./?], so we need to continue to attempt to
            # match this regexp as well as any successful paths that continue out of it.
            # both paths could be valid.
            if states = @regexp_states[s]
              slice_start = if previous_start.nil?
                start_index
              else
                previous_start
              end

              slice_length = end_index - slice_start
              curr_slice = full_string.slice(slice_start, slice_length)

              states.each { |re, v|
                # if we match, we can try moving past this
                next_states << [v, nil].freeze if !v.nil? && re.match?(curr_slice)
              }

              # and regardless, we must continue accepting tokens and retrying this regexp. we
              # need to remember where we started as well so we can take bigger slices.
              next_states << [s, slice_start].freeze
            end
          }

          next_states
        end

    def dom_id(record_or_class, prefix = nil)
      raise ArgumentError, "dom_id must be passed a record_or_class as the first argument, you passed #{record_or_class.inspect}" unless record_or_class

      record_id = record_key_for_dom_id(record_or_class) unless record_or_class.is_a?(Class)
      if record_id
        "#{dom_class(record_or_class, prefix)}#{JOIN}#{record_id}"
      else
        dom_class(record_or_class, prefix || NEW)
      end
          end

          {
            regexp_states:   simple_regexp,
            string_states:   @string_states,
            stdparam_states: @stdparam_states,
            accepting:       @accepting
          }
        end

      def xpath_internal(node, paths, handler, ns, binds)
        document = node.document
        return NodeSet.new(document) unless document

        if paths.length == 1
          return xpath_impl(node, paths.first, handler, ns, binds)
        end

        def visualizer(paths, title = "FSM")
          viz_dir   = File.join __dir__, "..", "visualizer"
          fsm_js    = File.read File.join(viz_dir, "fsm.js")
          fsm_css   = File.read File.join(viz_dir, "fsm.css")
          erb       = File.read File.join(viz_dir, "index.html.erb")
          states    = "function tt() { return #{to_json}; }"

          fun_routes = paths.sample(3).map do |ast|
            ast.filter_map { |n|
              case n
              when Nodes::Symbol
                case n.left
                when ":id" then rand(100).to_s
                when ":format" then %w{ xml json }.sample
                else
                  "omg"
                end
              when Nodes::Terminal then n.symbol
              else
                nil
              end
            }.join
          end

          stylesheets = [fsm_css]
          svg         = to_svg
          javascripts = [states, fsm_js]

          fun_routes  = fun_routes
          stylesheets = stylesheets
          svg         = svg
          javascripts = javascripts

          require "erb"
          template = ERB.new erb
          template.result(binding)
        end

        def []=(from, to, sym)
          to_mappings = states_hash_for(sym)[from] ||= {}
          case sym
          when Regexp
            # we must match the whole string to a token boundary
            if sym == DEFAULT_EXP
              sym = DEFAULT_EXP_ANCHORED
            else
              sym = /\A#{sym}\Z/
            end
          when Symbol
            # account for symbols in the constraints the same as strings
            sym = sym.to_s
          end
          to_mappings[sym] = to
        end


    def perform_now
      # Guard against jobs that were persisted before we started counting executions by zeroing out nil counters
      self.executions = (executions || 0) + 1

      deserialize_arguments_if_needed

      _perform_job
    rescue Exception => exception
      handled = rescue_with_handler(exception)
      return handled if handled

      run_after_discard_procs(exception)
      raise
    end

        private
            else
              raise ArgumentError, "unknown symbol: %s" % sym.class
            end
          end
      end
    end
  end
end
