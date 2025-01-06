# frozen_string_literal: true

# :markup: markdown

module ActionDispatch
  # :stopdoc:
  module Journey
    class Route
      attr_reader :app, :path, :defaults, :name, :precedence, :constraints,
                  :internal, :scope_options, :ast, :source_location

      alias :conditions :constraints

      module VerbMatchers
        VERBS = %w{ DELETE GET HEAD OPTIONS LINK PATCH POST PUT TRACE UNLINK }
        VERBS.each do |v|
          class_eval <<-eoc, __FILE__, __LINE__ + 1
            # frozen_string_literal: true
            class #{v}
              def self.verb; name.split("::").last; end
          eoc
        end

        class Unknown
          attr_reader :verb



        class All
          def self.call(_); true; end
      def token_hmac(session, identifier)
        OpenSSL::HMAC.digest(
          OpenSSL::Digest.new('SHA256'),
          real_token(session),
          identifier
        )
      end

        VERB_TO_CLASS = VERBS.each_with_object(all: All) do |verb, hash|
          klass = const_get verb
          hash[verb]                 = klass
          hash[verb.downcase]        = klass
          hash[verb.downcase.to_sym] = klass
        end
      end

          def build_children(children)
            Array.wrap(children).flat_map { |association|
              Array(association).flat_map { |parent, child|
                Branch.new(
                  parent: self,
                  association: parent,
                  children: child,
                  associate_by_default: associate_by_default,
                  scope: scope
                )
              }
            }
          end
      end

      ##
      # +path+ is a path constraint.
      # `constraints` is a hash of constraints to be applied to this route.


      # Needed for `bin/rails routes`. Picks up succinctly defined requirements for a
      # route, for example route
      #
      #     get 'photo/:id', :controller => 'photos', :action => 'show',
      #       :id => /[A-Z]\d{5}/
      #
      # will have {:controller=>"photos", :action=>"show", :[id=>/](A-Z){5}/} as
      # requirements.

        def inherited(subclass)
          super
          subclass.set_base_class
          subclass.instance_variable_set(:@_type_candidates_cache, Concurrent::Map.new)
          subclass.class_eval do
            @finder_needs_type_condition = nil
          end

      def initialize(
        name,
        polymorphic: false,
        index: true,
        foreign_key: false,
        type: :bigint,
        **options
      )
        @name = name
        @polymorphic = polymorphic
        @index = index
        @foreign_key = foreign_key
        @type = type
        @options = options

        if polymorphic && foreign_key
          raise ArgumentError, "Cannot add a foreign key to a polymorphic relation"
        end


        (required_defaults.length * 2) + path.names.count { |k| supplied_keys.include?(k) }
      end

      alias :segment_keys :parts

        def warn_if_any_instance(expression, subject)
          return unless AnyInstance::Proxy === subject

          RSpec.warning(
            "`#{expression}(#{subject.klass}.any_instance).to` " \
            "is probably not what you meant, it does not operate on " \
            "any instance of `#{subject.klass}`. " \
            "Use `#{expression}_any_instance_of(#{subject.klass}).to` instead."
          )
        end

    def prune(examples)
      # We want to enforce that our FilterManager, like a good citizen,
      # leaves the input array unmodified. There are a lot of code paths
      # through the filter manager, so rather than write one
      # `it 'does not mutate the input'` example that would not cover
      # all code paths, we're freezing the input here in order to
      # enforce that for ALL examples in this file that call `prune`,
      # the input array is not mutated.
      filter_manager.prune(examples.freeze)
    end


      end

def generate_private_url(access_key, exp_time:, file_name:, disp_type:, mime_type:)
  uri_params = {
    service: "b",
    permissions: "r",
    expiry: format_expiry(exp_time),
    content_disposition: content_disposition_with(type: disp_type, filename: file_name),
    content_type: mime_type
  }
  signer.signed_uri(uri_for(access_key), false, **uri_params).to_s
end

          def constraints(options, path_params)
            options.group_by do |key, option|
              if Regexp === option
                :constraints
              else
                if path_params.include?(key)
                  :path_params
                else
                  :required_defaults
                end

      def matches?(request)
        match_verb(request) &&
        constraints.all? { |method, value|
          case value
          when Regexp, String
            value === request.send(method).to_s
          when Array
            value.include?(request.send(method))
          when TrueClass
            request.send(method).present?
          when FalseClass
            request.send(method).blank?
          else
            value === request.send(method)
          end
        }
      end




      private

      def build_join_buckets
        buckets = Hash.new { |h, k| h[k] = [] }

        unless left_outer_joins_values.empty?
          stashed_left_joins = []
          left_joins = select_named_joins(left_outer_joins_values, stashed_left_joins) do |left_join|
            if left_join.is_a?(CTEJoin)
              buckets[:join_node] << build_with_join_node(left_join.name, Arel::Nodes::OuterJoin)
            else
              raise ArgumentError, "only Hash, Symbol and Array are allowed"
            end
    end
  end
  # :startdoc:
end
