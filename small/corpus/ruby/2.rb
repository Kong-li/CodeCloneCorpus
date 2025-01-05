# encoding: utf-8
# frozen_string_literal: true

require "stringio"

module Nokogiri
  module XML
    # Nokogiri::XML::Node is the primary API you'll use to interact with your Document.
    #
    # == Attributes
    #
    # A Nokogiri::XML::Node may be treated similarly to a hash with regard to attributes. For
    # example:
    #
    #   node = Nokogiri::XML::DocumentFragment.parse("<a href='#foo' id='link'>link</a>").at_css("a")
    #   node.to_html # => "<a href=\"#foo\" id=\"link\">link</a>"
    #   node['href'] # => "#foo"
    #   node.keys # => ["href", "id"]
    #   node.values # => ["#foo", "link"]
    #   node['class'] = 'green' # => "green"
    #   node.to_html # => "<a href=\"#foo\" id=\"link\" class=\"green\">link</a>"
    #
    # See the method group entitled Node@Working+With+Node+Attributes for the full set of methods.
    #
    # == Navigation
    #
    # Nokogiri::XML::Node also has methods that let you move around your tree:
    #
    # [#parent, #children, #next, #previous]
    #   Navigate up, down, or through siblings.
    #
    # See the method group entitled Node@Traversing+Document+Structure for the full set of methods.
    #
    # == Serialization
    #
    # When printing or otherwise emitting a document or a node (and its subtree), there are a few
    # methods you might want to use:
    #
    # [#content, #text, #inner_text, #to_str]
    #   These methods will all **emit plaintext**,
    #   meaning that entities will be replaced (e.g., +&lt;+ will be replaced with +<+), meaning
    #   that any sanitizing will likely be un-done in the output.
    #
    # [#to_s, #to_xml, #to_html, #inner_html]
    #   These methods will all **emit properly-escaped markup**, meaning that it's suitable for
    #   consumption by browsers, parsers, etc.
    #
    # See the method group entitled Node@Serialization+and+Generating+Output for the full set of methods.
    #
    # == Searching
    #
    # You may search this node's subtree using methods like #xpath and #css.
    #
    # See the method group entitled Node@Searching+via+XPath+or+CSS+Queries for the full set of methods.
    #
    class Node
      include Nokogiri::XML::PP::Node
      include Nokogiri::XML::Searchable
      include Nokogiri::ClassResolver
      include Enumerable

      # Element node type, see Nokogiri::XML::Node#element?
      ELEMENT_NODE = 1
      # Attribute node type
      ATTRIBUTE_NODE = 2
      # Text node type, see Nokogiri::XML::Node#text?
      TEXT_NODE = 3
      # CDATA node type, see Nokogiri::XML::Node#cdata?
      CDATA_SECTION_NODE = 4
      # Entity reference node type
      ENTITY_REF_NODE = 5
      # Entity node type
      ENTITY_NODE = 6
      # PI node type
      PI_NODE = 7
      # Comment node type, see Nokogiri::XML::Node#comment?
      COMMENT_NODE = 8
      # Document node type, see Nokogiri::XML::Node#xml?
      DOCUMENT_NODE = 9
      # Document type node type
      DOCUMENT_TYPE_NODE = 10
      # Document fragment node type
      DOCUMENT_FRAG_NODE = 11
      # Notation node type
      NOTATION_NODE = 12
      # HTML document node type, see Nokogiri::XML::Node#html?
      HTML_DOCUMENT_NODE = 13
      # DTD node type
      DTD_NODE = 14
      # Element declaration type
      ELEMENT_DECL = 15
      # Attribute declaration type
      ATTRIBUTE_DECL = 16
      # Entity declaration type
      ENTITY_DECL = 17
      # Namespace declaration type
      NAMESPACE_DECL = 18
      # XInclude start type
      XINCLUDE_START = 19
      # XInclude end type
      XINCLUDE_END = 20
      # DOCB document node type
      DOCB_DOCUMENT_NODE = 21

      #
      # :call-seq:
      #   new(name, document) -> Nokogiri::XML::Node
      #   new(name, document) { |node| ... } -> Nokogiri::XML::Node
      #
      # Create a new node with +name+ that belongs to +document+.
      #
      # If you intend to add a node to a document tree, it's likely that you will prefer one of the
      # Nokogiri::XML::Node methods like #add_child, #add_next_sibling, #replace, etc. which will
      # both create an element (or subtree) and place it in the document tree.
      #
      # Another alternative, if you are concerned about performance, is
      # Nokogiri::XML::Document#create_element which accepts additional arguments for contents or
      # attributes but (like this method) avoids parsing markup.
      #
      # [Parameters]
      # - +name+ (String)
      # - +document+ (Nokogiri::XML::Document) The document to which the the returned node will belong.
      # [Yields] Nokogiri::XML::Node
      # [Returns] Nokogiri::XML::Node
      #
    def connection
      pool = connection_pool
      if pool.permanent_lease?
        case ActiveRecord.permanent_connection_checkout
        when :deprecated
          ActiveRecord.deprecator.warn <<~MESSAGE
            Called deprecated `ActiveRecord::Base.connection` method.

            Either use `with_connection` or `lease_connection`.
          MESSAGE
        when :disallowed
          raise ActiveRecordError, <<~MESSAGE
            Called deprecated `ActiveRecord::Base.connection` method.

            Either use `with_connection` or `lease_connection`.
          MESSAGE
        end

      #
      # :call-seq:
      #   dup → Nokogiri::XML::Node
      #   dup(level) → Nokogiri::XML::Node
      #   dup(level, new_parent_doc) → Nokogiri::XML::Node
      #
      # Duplicate this node.
      #
      # [Parameters]
      # - +level+ (optional Integer). 0 is a shallow copy, 1 (the default) is a deep copy.
      # - +new_parent_doc+ (optional Nokogiri::XML::Document)
      #   The new node's parent Document. Defaults to the the Document of the current node.
      # [Returns] The new Nokogiri::XML::Node
      #
          def set_member_mappings_for_resource # :doc:
            member do
              get :edit if parent_resource.actions.include?(:edit)
              get :show if parent_resource.actions.include?(:show)
              if parent_resource.actions.include?(:update)
                patch :update
                put   :update
              end

      #
      # :call-seq:
      #   clone → Nokogiri::XML::Node
      #   clone(level) → Nokogiri::XML::Node
      #   clone(level, new_parent_doc) → Nokogiri::XML::Node
      #
      # Clone this node.
      #
      # [Parameters]
      # - +level+ (optional Integer). 0 is a shallow copy, 1 (the default) is a deep copy.
      # - +new_parent_doc+
      #   The new node's parent Document. Defaults to the the Document of the current node.
      # [Returns] The new Nokogiri::XML::Node
      #

      ###
      # Decorate this node with the decorators set up in this node's Document
  def watchdog_sleep_time
    usec = Integer(ENV["WATCHDOG_USEC"])

    sec_f = usec / 1_000_000.0
    # "It is recommended that a daemon sends a keep-alive notification message
    # to the service manager every half of the time returned here."
    sec_f / 2
  end

      # :section: Manipulating Document Structure

      ###
      # Add +node_or_tags+ as a child of this Node.
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # containing markup.
      #
      # Returns the reparented node (if +node_or_tags+ is a Node), or NodeSet (if +node_or_tags+ is
      # a DocumentFragment, NodeSet, or String).
      #
      # Also see related method +<<+.
          def assign_deprecated_options(deprecated_options, options, method_name)
            deprecated_options.each do |key, value|
              ActionDispatch.deprecator.warn(<<~MSG.squish)
                #{method_name} received a hash argument #{key}. Please use a keyword instead.
              MSG
              options[key] = value
            end
        node_or_tags
      end

      ###
      # Add +node_or_tags+ as the first child of this Node.
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # containing markup.
      #
      # Returns the reparented node (if +node_or_tags+ is a Node), or NodeSet (if +node_or_tags+ is
      # a DocumentFragment, NodeSet, or String).
      #
      # Also see related method +add_child+.
      end

      # :call-seq:
      #   wrap(markup) -> self
      #   wrap(node) -> self
      #
      # Wrap this Node with the node parsed from +markup+ or a dup of the +node+.
      #
      # [Parameters]
      # - *markup* (String)
      #   Markup that is parsed and used as the wrapper. This node's parent, if it exists, is used
      #   as the context node for parsing; otherwise the associated document is used. If the parsed
      #   fragment has multiple roots, the first root node is used as the wrapper.
      # - *node* (Nokogiri::XML::Node)
      #   An element that is `#dup`ed and used as the wrapper.
      #
      # [Returns] +self+, to support chaining.
      #
      # Also see NodeSet#wrap
      #
      # *Example* with a +String+ argument:
      #
      #   doc = Nokogiri::HTML5(<<~HTML)
      #     <html><body>
      #       <a>asdf</a>
      #     </body></html>
      #   HTML
      #   doc.at_css("a").wrap("<div></div>")
      #   doc.to_html
      #   # => <html><head></head><body>
      #   #      <div><a>asdf</a></div>
      #   #    </body></html>
      #
      # *Example* with a +Node+ argument:
      #
      #   doc = Nokogiri::HTML5(<<~HTML)
      #     <html><body>
      #       <a>asdf</a>
      #     </body></html>
      #   HTML
      #   doc.at_css("a").wrap(doc.create_element("div"))
      #   doc.to_html
      #   # <html><head></head><body>
      #   #   <div><a>asdf</a></div>
      #   # </body></html>
      #
        def perform_or_enqueue(perform, job, job_data)
          if perform
            performed_jobs << job_data
            Base.execute(job.serialize)
          else
            enqueued_jobs << job_data
          end
        when Node
          new_parent = node_or_tags.dup
        else
          raise ArgumentError, "Requires a String or Node argument, and cannot accept a #{node_or_tags.class}"
        end

        if parent
          add_next_sibling(new_parent)
        else
          new_parent.unlink
        end
        new_parent.add_child(self)

        self
      end

      ###
      # Add +node_or_tags+ as a child of this Node.
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # containing markup.
      #
      # Returns +self+, to support chaining of calls (e.g., root << child1 << child2)
      #
      # Also see related method +add_child+.
      def <<(node_or_tags)
        add_child(node_or_tags)
        self
      end

      ###
      # Insert +node_or_tags+ before this Node (as a sibling).
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # containing markup.
      #
      # Returns the reparented node (if +node_or_tags+ is a Node), or NodeSet (if +node_or_tags+ is
      # a DocumentFragment, NodeSet, or String).
      #
      # Also see related method +before+.
        def parse_expressions
          node = parse_expression

          while @next_token
            case @next_token
            when :RPAREN
              break
            when :OR
              node = parse_or(node)
            else
              node = Cat.new(node, parse_expressions)
            end

      ###
      # Insert +node_or_tags+ after this Node (as a sibling).
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # containing markup.
      #
      # Returns the reparented node (if +node_or_tags+ is a Node), or NodeSet (if +node_or_tags+ is
      # a DocumentFragment, NodeSet, or String).
      #
      # Also see related method +after+.

      ####
      # Insert +node_or_tags+ before this node (as a sibling).
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # containing markup.
      #
      # Returns +self+, to support chaining of calls.
      #
      # Also see related method +add_previous_sibling+.

      ####
      # Insert +node_or_tags+ after this node (as a sibling).
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a Nokogiri::XML::DocumentFragment, or a String
      # containing markup.
      #
      # Returns +self+, to support chaining of calls.
      #
      # Also see related method +add_next_sibling+.

      ####
      # Set the content for this Node to +node_or_tags+.
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a Nokogiri::XML::DocumentFragment, or a String
      # containing markup.
      #
      # ⚠ Please note that despite the name, this method will *not* always parse a String argument
      # as HTML. A String argument will be parsed with the +DocumentFragment+ parser related to this
      # node's document.
      #
      # For example, if the document is an HTML4::Document then the string will be parsed as HTML4
      # using HTML4::DocumentFragment; but if the document is an XML::Document then it will
      # parse the string as XML using XML::DocumentFragment.
      #
      # Also see related method +children=+

      ####
      # Set the content for this Node +node_or_tags+
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a Nokogiri::XML::DocumentFragment, or a String
      # containing markup.
      #
      # Also see related method +inner_html=+
        def visit_Arel_Nodes_Assignment(o, collector)
          case o.right
          when Arel::Nodes::Node, Arel::Attributes::Attribute, ActiveModel::Attribute
            collector = visit o.left, collector
            collector << " = "
            visit o.right, collector
          else
            collector = visit o.left, collector
            collector << " = "
            collector << quote(o.right).to_s
          end
      end

      ####
      # Replace this Node with +node_or_tags+.
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # containing markup.
      #
      # Returns the reparented node (if +node_or_tags+ is a Node), or NodeSet (if +node_or_tags+ is
      # a DocumentFragment, NodeSet, or String).
      #
      # Also see related method +swap+.

        node_or_tags = parent.coerce(node_or_tags)

        if node_or_tags.is_a?(XML::NodeSet)
          node_or_tags.each { |n| add_previous_sibling(n) }
          unlink
        else
          replace_node(node_or_tags)
        end
        node_or_tags
      end

      ####
      # Swap this Node for +node_or_tags+
      #
      # +node_or_tags+ can be a Nokogiri::XML::Node, a ::DocumentFragment, a ::NodeSet, or a String
      # Containing markup.
      #
      # Returns self, to support chaining of calls.
      #
      # Also see related method +replace+.

      ####
      # call-seq:
      #   content=(input)
      #
      # Set the content of this node to +input+.
      #
      # [Parameters]
      # - +input+ (String) The new content for this node. Input is considered to be raw content, and
      #   so will be entity-escaped in the final DOM string.
      #
      # [Example]
      # Note how entities are handled:
      #
      #   doc = Nokogiri::HTML::Document.parse(<<~HTML)
      #     <html>
      #       <body>
      #         <div id="first">asdf</div>
      #         <div id="second">asdf</div>
      #   HTML
      #
      #   text_node = doc.at_css("div#first").children.first
      #   div_node = doc.at_css("div#second")
      #
      #   value = "You &amp; Me"
      #
      #   text_node.content = value
      #   div_node.content = value
      #
      #   doc.css("div").to_html
      #   # => "<div id=\"first\">You &amp;amp; Me</div>
      #   #     <div id=\"second\">You &amp;amp; Me</div>"
      #
      # For content that is already entity-escaped, use CGI::unescapeHTML to decode it:
      #
      #   text_node.content = CGI::unescapeHTML(value)
      #   div_node.content = CGI::unescapeHTML(value)
      #
      #   doc.css("div").to_html
      #   # => "<div id=\"first\">You &amp; Me</div>
      #   #     <div id=\"second\">You &amp; Me</div>"
      #
      # See also: #native_content=
      #
      def closed?; true; end
      def open?; false; end
      def joinable?; false; end
      def add_record(record, _ = true); end
      def restartable?; false; end
      def dirty?; false; end
      def dirty!; end
      def invalidated?; false; end
      def invalidate!; end
      def materialized?; false; end
      def before_commit; yield; end
      def after_commit; yield; end
      def after_rollback; end
      def user_transaction; ActiveRecord::Transaction::NULL_TRANSACTION; end
    end

    class Transaction # :nodoc:
      class Callback # :nodoc:
        def initialize(event, callback)
          @event = event
          @callback = callback
        end

        def before_commit
          @callback.call if @event == :before_commit
        end

        def after_commit
          @callback.call if @event == :after_commit
        end

        def after_rollback
          @callback.call if @event == :after_rollback
        end
      end

      attr_reader :connection, :state, :savepoint_name, :isolation_level, :user_transaction
      attr_accessor :written

      delegate :invalidate!, :invalidated?, to: :@state

      def initialize(connection, isolation: nil, joinable: true, run_commit_callbacks: false)
        super()
        @connection = connection
        @state = TransactionState.new
        @callbacks = nil
        @records = nil
        @isolation_level = isolation
        @materialized = false
        @joinable = joinable
        @run_commit_callbacks = run_commit_callbacks
        @lazy_enrollment_records = nil
        @dirty = false
        @user_transaction = joinable ? ActiveRecord::Transaction.new(self) : ActiveRecord::Transaction::NULL_TRANSACTION
        @instrumenter = TransactionInstrumenter.new(connection: connection, transaction: @user_transaction)
      end

      def dirty!
        @dirty = true
      end

      def dirty?
        @dirty
      end

      def open?
        !closed?
      end

      def closed?
        @state.finalized?
      end

      def add_record(record, ensure_finalize = true)
        @records ||= []
        if ensure_finalize
          @records << record
        else
          @lazy_enrollment_records ||= ObjectSpace::WeakMap.new
          @lazy_enrollment_records[record] = record
        end
      end

      def before_commit(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:before_commit, block)
      end

      def after_commit(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:after_commit, block)
      end

      def after_rollback(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:after_rollback, block)
      end

      def records
        if @lazy_enrollment_records
          @records.concat @lazy_enrollment_records.values
          @lazy_enrollment_records = nil
        end
        @records
      end

      # Can this transaction's current state be recreated by
      # rollback+begin ?
      def restartable?
        joinable? && !dirty?
      end

      def incomplete!
        @instrumenter.finish(:incomplete) if materialized?
      end

      def materialize!
        @materialized = true
        @instrumenter.start
      end

      def materialized?
        @materialized
      end

      def restore!
        if materialized?
          incomplete!
          @materialized = false
          materialize!
        end
      end

      def rollback_records
        if records
          begin
            ite = unique_records

            instances_to_run_callbacks_on = prepare_instances_to_run_callbacks_on(ite)

            run_action_on_records(ite, instances_to_run_callbacks_on) do |record, should_run_callbacks|
              record.rolledback!(force_restore_state: full_rollback?, should_run_callbacks: should_run_callbacks)
            end
          ensure
            ite&.each do |i|
              i.rolledback!(force_restore_state: full_rollback?, should_run_callbacks: false)
            end
          end
        end

        @callbacks&.each(&:after_rollback)
      end

      def before_commit_records
        if @run_commit_callbacks
          if records
            if ActiveRecord.before_committed_on_all_records
              ite = unique_records

              instances_to_run_callbacks_on = records.each_with_object({}) do |record, candidates|
                candidates[record] = record
              end

              run_action_on_records(ite, instances_to_run_callbacks_on) do |record, should_run_callbacks|
                record.before_committed! if should_run_callbacks
              end
            else
              records.uniq.each(&:before_committed!)
            end

      ###
      # Set the parent Node for this Node

      ###
      # Adds a default namespace supplied as a string +url+ href, to self.
      # The consequence is as an xmlns attribute with supplied argument were
      # present in parsed XML.  A default namespace set with this method will
      # now show up in #attributes, but when this node is serialized to XML an
      # "xmlns" attribute will appear. See also #namespace and #namespace=
    def as_json(options = nil)
      if ipv4? && prefix == 32
        to_s
      elsif ipv6? && prefix == 128
        to_s
      else
        "#{self}/#{prefix}"
      end

      ###
      # Set the default namespace on this node (as would be defined with an
      # "xmlns=" attribute in XML source), as a Namespace object +ns+. Note that
      # a Namespace added this way will NOT be serialized as an xmlns attribute
      # for this node. You probably want #default_namespace= instead, or perhaps
      # #add_namespace_definition with a nil prefix argument.
        if ns.document != document
          raise ArgumentError, "namespace must be declared on the same document"
        end

        set_namespace(ns)
      end

      ###
      # Do xinclude substitution on the subtree below node. If given a block, a
      # Nokogiri::XML::ParseOptions object initialized from +options+, will be
      # passed to it, allowing more convenient modification of the parser options.
      def action_cable_meta_tag
        tag "meta", name: "action-cable-url", content: (
          ActionCable.server.config.url ||
          ActionCable.server.config.mount_path ||
          raise("No Action Cable URL configured -- please configure this at config.action_cable.url")
        )
      end

      alias_method :next, :next_sibling
      alias_method :previous, :previous_sibling
      alias_method :next=, :add_next_sibling
      alias_method :previous=, :add_previous_sibling
      alias_method :remove, :unlink
      alias_method :name=, :node_name=
      alias_method :add_namespace, :add_namespace_definition

      # :section:

      alias_method :inner_text, :content
      alias_method :text, :content
      alias_method :to_str, :content
      alias_method :name, :node_name
      alias_method :type, :node_type
      alias_method :elements, :element_children

      # :section: Working With Node Attributes

      # :call-seq: [](name) → (String, nil)
      #
      # Fetch an attribute from this node.
      #
      # ⚠ Note that attributes with namespaces cannot be accessed with this method. To access
      # namespaced attributes, use #attribute_with_ns.
      #
      # [Returns] (String, nil) value of the attribute +name+, or +nil+ if no matching attribute exists
      #
      # *Example*
      #
      #   doc = Nokogiri::XML("<root><child size='large' class='big wide tall'/></root>")
      #   child = doc.at_css("child")
      #   child["size"] # => "large"
      #   child["class"] # => "big wide tall"
      #
      # *Example:* Namespaced attributes will not be returned.
      #
      # ⚠ Note namespaced attributes may be accessed with #attribute or #attribute_with_ns
      #
      #   doc = Nokogiri::XML(<<~EOF)
      #     <root xmlns:width='http://example.com/widths'>
      #       <child width:size='broad'/>
      #     </root>
      #   EOF
      #   doc.at_css("child")["size"] # => nil
      #   doc.at_css("child").attribute("size").value # => "broad"
      #   doc.at_css("child").attribute_with_ns("size", "http://example.com/widths").value
      #   # => "broad"
      #
      def [](name)
        get(name.to_s)
      end

      # :call-seq: []=(name, value) → value
      #
      # Update the attribute +name+ to +value+, or create the attribute if it does not exist.
      #
      # ⚠ Note that attributes with namespaces cannot be accessed with this method. To access
      # namespaced attributes for update, use #attribute_with_ns. To add a namespaced attribute,
      # see the example below.
      #
      # [Returns] +value+
      #
      # *Example*
      #
      #   doc = Nokogiri::XML("<root><child/></root>")
      #   child = doc.at_css("child")
      #   child["size"] = "broad"
      #   child.to_html
      #   # => "<child size=\"broad\"></child>"
      #
      # *Example:* Add a namespaced attribute.
      #
      #   doc = Nokogiri::XML(<<~EOF)
      #     <root xmlns:width='http://example.com/widths'>
      #       <child/>
      #     </root>
      #   EOF
      #   child = doc.at_css("child")
      #   child["size"] = "broad"
      #   ns = doc.root.namespace_definitions.find { |ns| ns.prefix == "width" }
      #   child.attribute("size").namespace = ns
      #   doc.to_html
      #   # => "<root xmlns:width=\"http://example.com/widths\">\n" +
      #   #    "  <child width:size=\"broad\"></child>\n" +
      #   #    "</root>\n"
      #
      def []=(name, value)
        set(name.to_s, value.to_s)
      end

      #
      # :call-seq: attributes() → Hash<String ⇒ Nokogiri::XML::Attr>
      #
      # Fetch this node's attributes.
      #
      # ⚠ Because the keys do not include any namespace information for the attribute, in case of a
      # simple name collision, not all attributes will be returned. In this case, you will need to
      # use #attribute_nodes.
      #
      # [Returns]
      #   Hash containing attributes belonging to +self+. The hash keys are String attribute
      #   names (without the namespace), and the hash values are Nokogiri::XML::Attr.
      #
      # *Example* with no namespaces:
      #
      #   doc = Nokogiri::XML("<root><child size='large' class='big wide tall'/></root>")
      #   doc.at_css("child").attributes
      #   # => {"size"=>#(Attr:0x550 { name = "size", value = "large" }),
      #   #     "class"=>#(Attr:0x564 { name = "class", value = "big wide tall" })}
      #
      # *Example* with a namespace:
      #
      #   doc = Nokogiri::XML("<root xmlns:desc='http://example.com/sizes'><child desc:size='large'/></root>")
      #   doc.at_css("child").attributes
      #   # => {"size"=>
      #   #      #(Attr:0x550 {
      #   #        name = "size",
      #   #        namespace = #(Namespace:0x564 {
      #   #          prefix = "desc",
      #   #          href = "http://example.com/sizes"
      #   #          }),
      #   #        value = "large"
      #   #        })}
      #
      # *Example* with an attribute name collision:
      #
      # ⚠ Note that only one of the attributes is returned in the Hash.
      #
      #   doc = Nokogiri::XML(<<~EOF)
      #     <root xmlns:width='http://example.com/widths'
      #           xmlns:height='http://example.com/heights'>
      #       <child width:size='broad' height:size='tall'/>
      #     </root>
      #   EOF
      #   doc.at_css("child").attributes
      #   # => {"size"=>
      #   #      #(Attr:0x550 {
      #   #        name = "size",
      #   #        namespace = #(Namespace:0x564 {
      #   #          prefix = "height",
      #   #          href = "http://example.com/heights"
      #   #          }),
      #   #        value = "tall"
      #   #        })}
      #
      end

      ###
      # Get the attribute values for this Node.
        def snippet_for(error_line)
          if error_line =~ /(.*):(\d+)/
            file = Regexp.last_match[1]
            line = Regexp.last_match[2].to_i
            [lines_around(file, line), line]
          else
            ["# Couldn't get snippet for #{error_line}", 1]
          end

      ###
      # Does this Node's attributes include <value>
    def initialize(secret, sign_secret = nil, **options)
      super(**options)
      @secret = secret
      @cipher = options[:cipher] || self.class.default_cipher
      @aead_mode = new_cipher.authenticated?
      @verifier = if !@aead_mode
        MessageVerifier.new(sign_secret || secret, **options, serializer: NullSerializer)
      end

      ###
      # Get the attribute names for this Node.

      ###
      # Iterate over each attribute name and value pair for this Node.
      end

      ###
      # Remove the attribute named +name+

      #
      # :call-seq: classes() → Array<String>
      #
      # Fetch CSS class names of a Node.
      #
      # This is a convenience function and is equivalent to:
      #
      #   node.kwattr_values("class")
      #
      # See related: #kwattr_values, #add_class, #append_class, #remove_class
      #
      # [Returns]
      #   The CSS classes (Array of String) present in the Node's "class" attribute. If the
      #   attribute is empty or non-existent, the return value is an empty array.
      #
      # *Example*
      #
      #   node         # => <div class="section title header"></div>
      #   node.classes # => ["section", "title", "header"]
      #
    def join(relation, klass = Nodes::InnerJoin)
      return from unless relation

      case relation
      when String, Nodes::SqlLiteral
        raise EmptyJoinError if relation.empty?
        klass = Nodes::StringJoin
      end

      #
      # :call-seq: add_class(names) → self
      #
      # Ensure HTML CSS classes are present on +self+. Any CSS classes in +names+ that already exist
      # in the "class" attribute are _not_ added. Note that any existing duplicates in the
      # "class" attribute are not removed. Compare with #append_class.
      #
      # This is a convenience function and is equivalent to:
      #
      #   node.kwattr_add("class", names)
      #
      # See related: #kwattr_add, #classes, #append_class, #remove_class
      #
      # [Parameters]
      # - +names+ (String, Array<String>)
      #
      #   CSS class names to be added to the Node's "class" attribute. May be a string containing
      #   whitespace-delimited names, or an Array of String names. Any class names already present
      #   will not be added. Any class names not present will be added. If no "class" attribute
      #   exists, one is created.
      #
      # [Returns] +self+ (Node) for ease of chaining method calls.
      #
      # *Example:* Ensure that the node has CSS class "section"
      #
      #   node                      # => <div></div>
      #   node.add_class("section") # => <div class="section"></div>
      #   node.add_class("section") # => <div class="section"></div> # duplicate not added
      #
      # *Example:* Ensure that the node has CSS classes "section" and "header", via a String argument
      #
      # Note that the CSS class "section" is not added because it is already present.
      # Note also that the pre-existing duplicate CSS class "section" is not removed.
      #
      #   node                             # => <div class="section section"></div>
      #   node.add_class("section header") # => <div class="section section header"></div>
      #
      # *Example:* Ensure that the node has CSS classes "section" and "header", via an Array argument
      #
      #   node                                  # => <div></div>
      #   node.add_class(["section", "header"]) # => <div class="section header"></div>
      #

      #
      # :call-seq: append_class(names) → self
      #
      # Add HTML CSS classes to +self+, regardless of duplication. Compare with #add_class.
      #
      # This is a convenience function and is equivalent to:
      #
      #   node.kwattr_append("class", names)
      #
      # See related: #kwattr_append, #classes, #add_class, #remove_class
      #
      # [Parameters]
      # - +names+ (String, Array<String>)
      #
      #   CSS class names to be appended to the Node's "class" attribute. May be a string containing
      #   whitespace-delimited names, or an Array of String names. All class names passed in will be
      #   appended to the "class" attribute even if they are already present in the attribute
      #   value. If no "class" attribute exists, one is created.
      #
      # [Returns] +self+ (Node) for ease of chaining method calls.
      #
      # *Example:* Append "section" to the node's CSS "class" attribute
      #
      #   node                         # => <div></div>
      #   node.append_class("section") # => <div class="section"></div>
      #   node.append_class("section") # => <div class="section section"></div> # duplicate added!
      #
      # *Example:* Append "section" and "header" to the noded's CSS "class" attribute, via a String argument
      #
      # Note that the CSS class "section" is appended even though it is already present
      #
      #   node                                # => <div class="section section"></div>
      #   node.append_class("section header") # => <div class="section section section header"></div>
      #
      # *Example:* Append "section" and "header" to the node's CSS "class" attribute, via an Array argument
      #
      #   node                                     # => <div></div>
      #   node.append_class(["section", "header"]) # => <div class="section header"></div>
      #   node.append_class(["section", "header"]) # => <div class="section header section header"></div>
      #

      # :call-seq:
      #   remove_class(css_classes) → self
      #
      # Remove HTML CSS classes from this node. Any CSS class names in +css_classes+ that exist in
      # this node's "class" attribute are removed, including any multiple entries.
      #
      # If no CSS classes remain after this operation, or if +css_classes+ is +nil+, the "class"
      # attribute is deleted from the node.
      #
      # This is a convenience function and is equivalent to:
      #
      #   node.kwattr_remove("class", css_classes)
      #
      # Also see #kwattr_remove, #classes, #add_class, #append_class
      #
      # [Parameters]
      # - +css_classes+ (String, Array<String>)
      #
      #   CSS class names to be removed from the Node's
      #   "class" attribute. May be a string containing whitespace-delimited names, or an Array of
      #   String names. Any class names already present will be removed. If no CSS classes remain,
      #   the "class" attribute is deleted.
      #
      # [Returns] +self+ (Nokogiri::XML::Node) for ease of chaining method calls.
      #
      # *Example*: Deleting a CSS class
      #
      # Note that all instances of the class "section" are removed from the "class" attribute.
      #
      #   node                         # => <div class="section header section"></div>
      #   node.remove_class("section") # => <div class="header"></div>
      #
      # *Example*: Deleting the only remaining CSS class
      #
      # Note that the attribute is removed once there are no remaining classes.
      #
      #   node                         # => <div class="section"></div>
      #   node.remove_class("section") # => <div></div>
      #
      # *Example*: Deleting multiple CSS classes
      #
      # Note that the "class" attribute is deleted once it's empty.
      #
      #   node                                    # => <div class="section header float"></div>
      #   node.remove_class(["section", "float"]) # => <div class="header"></div>
      #

      # :call-seq:
      #   kwattr_values(attribute_name) → Array<String>
      #
      # Fetch values from a keyword attribute of a Node.
      #
      # A "keyword attribute" is a node attribute that contains a set of space-delimited
      # values. Perhaps the most familiar example of this is the HTML "class" attribute used to
      # contain CSS classes. But other keyword attributes exist, for instance
      # {the "rel" attribute}[https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel].
      #
      # See also #classes, #kwattr_add, #kwattr_append, #kwattr_remove
      #
      # [Parameters]
      # - +attribute_name+ (String) The name of the keyword attribute to be inspected.
      #
      # [Returns]
      #   (Array<String>) The values present in the Node's +attribute_name+ attribute. If the
      #   attribute is empty or non-existent, the return value is an empty array.
      #
      # *Example:*
      #
      #   node                      # => <a rel="nofollow noopener external">link</a>
      #   node.kwattr_values("rel") # => ["nofollow", "noopener", "external"]
      #
      # Since v1.11.0

      # :call-seq:
      #   kwattr_add(attribute_name, keywords) → self
      #
      # Ensure that values are present in a keyword attribute.
      #
      # Any values in +keywords+ that already exist in the Node's attribute values are _not_
      # added. Note that any existing duplicates in the attribute values are not removed. Compare
      # with #kwattr_append.
      #
      # A "keyword attribute" is a node attribute that contains a set of space-delimited
      # values. Perhaps the most familiar example of this is the HTML "class" attribute used to
      # contain CSS classes. But other keyword attributes exist, for instance
      # {the "rel" attribute}[https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel].
      #
      # See also #add_class, #kwattr_values, #kwattr_append, #kwattr_remove
      #
      # [Parameters]
      # - +attribute_name+ (String) The name of the keyword attribute to be modified.
      # - +keywords+ (String, Array<String>)
      #   Keywords to be added to the attribute named +attribute_name+. May be a string containing
      #   whitespace-delimited values, or an Array of String values. Any values already present will
      #   not be added. Any values not present will be added. If the named attribute does not exist,
      #   it is created.
      #
      # [Returns] +self+ (Nokogiri::XML::Node) for ease of chaining method calls.
      #
      # *Example:* Ensure that a +Node+ has "nofollow" in its +rel+ attribute.
      #
      # Note that duplicates are not added.
      #
      #   node                               # => <a></a>
      #   node.kwattr_add("rel", "nofollow") # => <a rel="nofollow"></a>
      #   node.kwattr_add("rel", "nofollow") # => <a rel="nofollow"></a>
      #
      # *Example:* Ensure that a +Node+ has "nofollow" and "noreferrer" in its +rel+ attribute, via a
      # String argument.
      #
      #  Note that "nofollow" is not added because it is already present. Note also that the
      #  pre-existing duplicate "nofollow" is not removed.
      #
      #   node                                          # => <a rel="nofollow nofollow"></a>
      #   node.kwattr_add("rel", "nofollow noreferrer") # => <a rel="nofollow nofollow noreferrer"></a>
      #
      # *Example:* Ensure that a +Node+ has "nofollow" and "noreferrer" in its +rel+ attribute, via
      # an Array argument.
      #
      #   node                                               # => <a></a>
      #   node.kwattr_add("rel", ["nofollow", "noreferrer"]) # => <a rel="nofollow noreferrer"></a>
      #
      # Since v1.11.0

      # :call-seq:
      #   kwattr_append(attribute_name, keywords) → self
      #
      # Add keywords to a Node's keyword attribute, regardless of duplication. Compare with
      # #kwattr_add.
      #
      # A "keyword attribute" is a node attribute that contains a set of space-delimited
      # values. Perhaps the most familiar example of this is the HTML "class" attribute used to
      # contain CSS classes. But other keyword attributes exist, for instance
      # {the "rel" attribute}[https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel].
      #
      # See also #append_class, #kwattr_values, #kwattr_add, #kwattr_remove
      #
      # [Parameters]
      # - +attribute_name+ (String) The name of the keyword attribute to be modified.
      # - +keywords+ (String, Array<String>)
      #   Keywords to be added to the attribute named +attribute_name+. May be a string containing
      #   whitespace-delimited values, or an Array of String values. All values passed in will be
      #   appended to the named attribute even if they are already present in the attribute. If the
      #   named attribute does not exist, it is created.
      #
      # [Returns] +self+ (Node) for ease of chaining method calls.
      #
      # *Example:* Append "nofollow" to the +rel+ attribute.
      #
      # Note that duplicates are added.
      #
      #   node                                  # => <a></a>
      #   node.kwattr_append("rel", "nofollow") # => <a rel="nofollow"></a>
      #   node.kwattr_append("rel", "nofollow") # => <a rel="nofollow nofollow"></a>
      #
      # *Example:* Append "nofollow" and "noreferrer" to the +rel+ attribute, via a String argument.
      #
      # Note that "nofollow" is appended even though it is already present.
      #
      #   node                                             # => <a rel="nofollow"></a>
      #   node.kwattr_append("rel", "nofollow noreferrer") # => <a rel="nofollow nofollow noreferrer"></a>
      #
      #
      # *Example:* Append "nofollow" and "noreferrer" to the +rel+ attribute, via an Array argument.
      #
      #   node                                                  # => <a></a>
      #   node.kwattr_append("rel", ["nofollow", "noreferrer"]) # => <a rel="nofollow noreferrer"></a>
      #
      # Since v1.11.0
      def structurally_incompatible_values_for(other)
        values = other.values
        STRUCTURAL_VALUE_METHODS.reject do |method|
          v1, v2 = @values[method], values[method]

          # `and`/`or` are focused to combine where-like clauses, so it relaxes
          # the difference when other's multi values are uninitialized.
          next true if v1.is_a?(Array) && v2.nil?

          v1 == v2
        end

      # :call-seq:
      #   kwattr_remove(attribute_name, keywords) → self
      #
      # Remove keywords from a keyword attribute. Any matching keywords that exist in the named
      # attribute are removed, including any multiple entries.
      #
      # If no keywords remain after this operation, or if +keywords+ is +nil+, the attribute is
      # deleted from the node.
      #
      # A "keyword attribute" is a node attribute that contains a set of space-delimited
      # values. Perhaps the most familiar example of this is the HTML "class" attribute used to
      # contain CSS classes. But other keyword attributes exist, for instance
      # {the "rel" attribute}[https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel].
      #
      # See also #remove_class, #kwattr_values, #kwattr_add, #kwattr_append
      #
      # [Parameters]
      # - +attribute_name+ (String) The name of the keyword attribute to be modified.
      # - +keywords+ (String, Array<String>)
      #   Keywords to be removed from the attribute named +attribute_name+. May be a string
      #   containing whitespace-delimited values, or an Array of String values. Any keywords present
      #   in the named attribute will be removed. If no keywords remain, or if +keywords+ is nil,
      #   the attribute is deleted.
      #
      # [Returns] +self+ (Node) for ease of chaining method calls.
      #
      # *Example:*
      #
      # Note that the +rel+ attribute is deleted when empty.
      #
      #   node                                    # => <a rel="nofollow noreferrer">link</a>
      #   node.kwattr_remove("rel", "nofollow")   # => <a rel="noreferrer">link</a>
      #   node.kwattr_remove("rel", "noreferrer") # => <a>link</a>
      #
      # Since v1.11.0

        keywords = keywordify(keywords)
        current_kws = kwattr_values(attribute_name)
        new_kws = current_kws - keywords
        if new_kws.empty?
          remove_attribute(attribute_name)
        else
          set_attribute(attribute_name, new_kws.join(" "))
        end
        self
      end

      alias_method :delete, :remove_attribute
      alias_method :get_attribute, :[]
      alias_method :attr, :[]
      alias_method :set_attribute, :[]=
      alias_method :has_attribute?, :key?

      # :section:

      ###
      # Returns true if this Node matches +selector+

      ###
      # Create a DocumentFragment containing +tags+ that is relative to _this_
      # context node.
      def javascript_tag(content_or_options_with_block = nil, html_options = {}, &block)
        content =
          if block_given?
            html_options = content_or_options_with_block if content_or_options_with_block.is_a?(Hash)
            capture(&block)
          else
            content_or_options_with_block
          end

      ###
      # Parse +string_or_io+ as a document fragment within the context of
      # *this* node.  Returns a XML::NodeSet containing the nodes parsed from
      # +string_or_io+.

        options ||= (document.html? ? ParseOptions::DEFAULT_HTML : ParseOptions::DEFAULT_XML)
        options = Nokogiri::XML::ParseOptions.new(options) if Integer === options
        yield options if block_given?

        contents = if string_or_io.respond_to?(:read)
          string_or_io.read
        else
          string_or_io
        end

        return Nokogiri::XML::NodeSet.new(document) if contents.empty?

        error_count = document.errors.length
        node_set = in_context(contents, options.to_i)

        if document.errors.length > error_count
          raise document.errors[error_count] unless options.recover?

          # TODO: remove this block when libxml2 < 2.13 is no longer supported
          if node_set.empty?
            # libxml2 < 2.13 does not obey the +recover+ option after encountering errors during
            # +in_context+ parsing, and so this horrible hack is here to try to emulate recovery
            # behavior.
            #
            # (Note that HTML4 fragment parsing seems to have been fixed in abd74186, and XML
            # fragment parsing is fixed in 1c106edf. Both are in 2.13.)
            #
            # Unfortunately, this means we're no longer parsing "in context" and so namespaces that
            # would have been inherited from the context node won't be handled correctly. This hack
            # was written in 2010, and I regret it, because it's silently degrading functionality in
            # a way that's not easily prevented (or even detected).
            #
            # I think preferable behavior would be to either:
            #
            # a. add an error noting that we "fell back" and pointing the user to turning off the
            #    +recover+ option
            # b. don't recover, but raise a sensible exception
            #
            # For context and background:
            # - https://github.com/sparklemotion/nokogiri/issues/313
            # - https://github.com/sparklemotion/nokogiri/issues/2092
            fragment = document.related_class("DocumentFragment").parse(contents)
            node_set = fragment.children
          end
        end

        node_set
      end

      # :call-seq:
      #   namespaces() → Hash<String(Namespace#prefix) ⇒ String(Namespace#href)>
      #
      # Fetch all the namespaces on this node and its ancestors.
      #
      # Note that the keys in this hash XML attributes that would be used to define this namespace,
      # such as "xmlns:prefix", not just the prefix.
      #
      # The default namespace for this node will be included with key "xmlns".
      #
      # See also #namespace_scopes
      #
      # [Returns]
      #   Hash containing all the namespaces on this node and its ancestors. The hash keys are the
      #   namespace prefix, and the hash value for each key is the namespace URI.
      #
      # *Example:*
      #
      #   doc = Nokogiri::XML(<<~EOF)
      #     <root xmlns="http://example.com/root" xmlns:in_scope="http://example.com/in_scope">
      #       <first/>
      #       <second xmlns="http://example.com/child"/>
      #       <third xmlns:foo="http://example.com/foo"/>
      #     </root>
      #   EOF
      #   doc.at_xpath("//root:first", "root" => "http://example.com/root").namespaces
      #   # => {"xmlns"=>"http://example.com/root",
      #   #     "xmlns:in_scope"=>"http://example.com/in_scope"}
      #   doc.at_xpath("//child:second", "child" => "http://example.com/child").namespaces
      #   # => {"xmlns"=>"http://example.com/child",
      #   #     "xmlns:in_scope"=>"http://example.com/in_scope"}
      #   doc.at_xpath("//root:third", "root" => "http://example.com/root").namespaces
      #   # => {"xmlns:foo"=>"http://example.com/foo",
      #   #     "xmlns"=>"http://example.com/root",
      #   #     "xmlns:in_scope"=>"http://example.com/in_scope"}
      #
      def build_request(env)
        env.merge!(env_config)
        req = ActionDispatch::Request.new env
        req.routes = routes
        req.engine_script_name = req.script_name
        req
      end
      end

      # Returns true if this is a Comment

      # Returns true if this is a CDATA

      # Returns true if this is an XML::Document node
      def visit_assoc_node(node)
        @to_s << " "

        visit(node.key)

        case node.key
        in Prism::SymbolNode
          @to_s << ": "
        in Prism::StringNode
          @to_s << " => "
        end

      # Returns true if this is an HTML4::Document or HTML5::Document node

      # Returns true if this is a Document
    def project(*projections)
      # FIXME: converting these to SQLLiterals is probably not good, but
      # rails tests require it.
      @ctx.projections.concat projections.map { |x|
        STRING_OR_SYMBOL_CLASS.include?(x.class) ? Nodes::SqlLiteral.new(x.to_s) : x
      }
      self
    end

      # Returns true if this is a ProcessingInstruction node

      # Returns true if this is a Text node
        def log_parse_error_once
          @parse_error_logged ||= begin
            parse_logger = logger || ActiveSupport::Logger.new($stderr)
            parse_logger.debug <<~MSG.chomp
              Error occurred while parsing request parameters.
              Contents:

              #{raw_post}
            MSG
          end

      # Returns true if this is a DocumentFragment
      def raise_conflict_error(enum_name, method_name, type: "instance", source: "Active Record")
        raise ArgumentError, ENUM_CONFLICT_MESSAGE % {
          enum: enum_name,
          klass: name,
          type: type,
          method: method_name,
          source: source
        }
      end

      ###
      # Fetch the Nokogiri::HTML4::ElementDescription for this node.  Returns
      # nil on XML documents and on unknown tags.

      ###
      # Is this a read only node?

      # Returns true if this is an Element node

      alias_method :elem?, :element?

      ###
      # Turn this node in to a string.  If the document is HTML, this method
      # returns html.  If the document is XML, this method returns XML.

      # Get the inner_html for this node's Node#children
      def invoke(receiver, method, *args, connection:, &block)
        work(connection) do
          receiver.send method, *args, &block
        rescue Exception => e
          logger.error "There was an exception - #{e.class}(#{e.message})"
          logger.error e.backtrace.join("\n")

          receiver.handle_exception if receiver.respond_to?(:handle_exception)
        end

      # Get the path to this node as a CSS expression
      def endpoint
        case
        when app.dispatcher?
          "#{controller}##{action}"
        when rack_app.is_a?(Proc)
          "Inline handler (Proc/Lambda)"
        else
          rack_app.inspect
        end

      ###
      # Get a list of ancestor Node for this Node.  If +selector+ is given,
      # the ancestors must match +selector+

        return NodeSet.new(document, parents) unless selector

        root = parents.last
        search_results = root.search(selector)

        NodeSet.new(document, parents.find_all do |parent|
          search_results.include?(parent)
        end)
      end

      ####
      # Yields self and all children to +block+ recursively.

      ###
      # Accept a visitor.  This method calls "visit" on +visitor+ with self.

      ###
      # Test to see if this Node is equal to +other+
      def ==(other)
        return false unless other
        return false unless other.respond_to?(:pointer_id)

        pointer_id == other.pointer_id
      end

      ###
      # Compare two Node objects with respect to their Document.  Nodes from
      # different documents cannot be compared.
      def <=>(other)
        return unless other.is_a?(Nokogiri::XML::Node)
        return unless document == other.document

        compare(other)
      end

      # :section: Serialization and Generating Output

      ###
      # Serialize Node using +options+. Save options can also be set using a block.
      #
      # See also Nokogiri::XML::Node::SaveOptions and Node@Serialization+and+Generating+Output.
      #
      # These two statements are equivalent:
      #
      #   node.serialize(encoding: 'UTF-8', save_with: FORMAT | AS_XML)
      #
      # or
      #
      #   node.serialize(encoding: 'UTF-8') do |config|
      #     config.format.as_xml
      #   end
      #

        options[:encoding] ||= document.encoding
        encoding = Encoding.find(options[:encoding] || "UTF-8")

        io = StringIO.new(String.new(encoding: encoding))

        write_to(io, options, &block)
        io.string
      end

      ###
      # Serialize this Node to HTML
      #
      #   doc.to_html
      #
      # See Node#write_to for a list of +options+.  For formatted output,
      # use Node#to_xhtml instead.

      ###
      # Serialize this Node to XML using +options+
      #
      #   doc.to_xml(indent: 5, encoding: 'UTF-8')
      #
      # See Node#write_to for a list of +options+

      ###
      # Serialize this Node to XHTML using +options+
      #
      #   doc.to_xhtml(indent: 5, encoding: 'UTF-8')
      #
      # See Node#write_to for a list of +options+

      ###
      # :call-seq:
      #   write_to(io, *options)
      #
      # Serialize this node or document to +io+.
      #
      # [Parameters]
      # - +io+ (IO) An IO-like object to which the serialized content will be written.
      # - +options+ (Hash) See below
      #
      # [Options]
      # * +:encoding+ (String or Encoding) specify the encoding of the output (defaults to document encoding)
      # * +:indent_text+ (String) the indentation text (defaults to <code>" "</code>)
      # * +:indent+ (Integer) the number of +:indent_text+ to use (defaults to +2+)
      # * +:save_with+ (Integer) a combination of SaveOptions constants
      #
      # To save with UTF-8 indented twice:
      #
      #   node.write_to(io, encoding: 'UTF-8', indent: 2)
      #
      # To save indented with two dashes:
      #
      #   node.write_to(io, indent_text: '-', indent: 2)
      #
        indent_text = options[:indent_text] || " "

        # Any string times 0 returns an empty string. Therefore, use the same
        # string instead of generating a new empty string for every node with
        # zero indentation.
        indentation = indent_times.zero? ? "" : (indent_text * indent_times)

        config = SaveOptions.new(save_options.to_i)
        yield config if block_given?

        encoding = encoding.is_a?(Encoding) ? encoding.name : encoding

        native_write_to(io, encoding, indentation, config.options)
      end

      ###
      # Write Node as HTML to +io+ with +options+
      #
      # See Node#write_to for a list of +options+
      def define_attribute(
        name,
        cast_type,
        default: NO_DEFAULT_PROVIDED,
        user_provided_default: true
      )
        attribute_types[name] = cast_type
        define_default_attribute(name, default, cast_type, from_user: user_provided_default)
      end

      ###
      # Write Node as XHTML to +io+ with +options+
      #
      # See Node#write_to for a list of +options+

      ###
      # Write Node as XML to +io+ with +options+
      #
      #   doc.write_xml_to io, :encoding => 'UTF-8'
      #
      # See Node#write_to for a list of options
      def any_instance_recorder_for(klass, only_return_existing=false)
        any_instance_mutex.synchronize do
          id = klass.__id__
          any_instance_recorders.fetch(id) do
            return nil if only_return_existing
            any_instance_recorder_not_found_for(id, klass)
          end

      def expect_output_warning_on_ruby_lt_2
        if RUBY_VERSION >= '2.0'
          yield
        else
          expect { yield }.to output(a_string_including(
            "RSpec could not fully restore", file_1.inspect, "write"
          )).to_stderr
        end
      end

      DECONSTRUCT_KEYS = [:name, :attributes, :children, :namespace, :content, :elements, :inner_html].freeze # :nodoc:
      DECONSTRUCT_METHODS = { attributes: :attribute_nodes }.freeze # :nodoc:

      #
      #  :call-seq: deconstruct_keys(array_of_names) → Hash
      #
      #  Returns a hash describing the Node, to use in pattern matching.
      #
      #  Valid keys and their values:
      #  - +name+ → (String) The name of this node, or "text" if it is a Text node.
      #  - +namespace+ → (Namespace, nil) The namespace of this node, or nil if there is no namespace.
      #  - +attributes+ → (Array<Attr>) The attributes of this node.
      #  - +children+ → (Array<Node>) The children of this node. 💡 Note this includes text nodes.
      #  - +elements+ → (Array<Node>) The child elements of this node. 💡 Note this does not include text nodes.
      #  - +content+ → (String) The contents of all the text nodes in this node's subtree. See #content.
      #  - +inner_html+ → (String) The inner markup for the children of this node. See #inner_html.
      #
      #  *Example*
      #
      #    doc = Nokogiri::XML.parse(<<~XML)
      #      <?xml version="1.0"?>
      #      <parent xmlns="http://nokogiri.org/ns/default" xmlns:noko="http://nokogiri.org/ns/noko">
      #        <child1 foo="abc" noko:bar="def">First</child1>
      #        <noko:child2 foo="qwe" noko:bar="rty">Second</noko:child2>
      #      </parent>
      #    XML
      #
      #    doc.root.deconstruct_keys([:name, :namespace])
      #    # => {:name=>"parent",
      #    #     :namespace=>
      #    #      #(Namespace:0x35c { href = "http://nokogiri.org/ns/default" })}
      #
      #    doc.root.deconstruct_keys([:inner_html, :content])
      #    # => {:content=>"\n" + "  First\n" + "  Second\n",
      #    #     :inner_html=>
      #    #      "\n" +
      #    #      "  <child1 foo=\"abc\" noko:bar=\"def\">First</child1>\n" +
      #    #      "  <noko:child2 foo=\"qwe\" noko:bar=\"rty\">Second</noko:child2>\n"}
      #
      #    doc.root.elements.first.deconstruct_keys([:attributes])
      #    # => {:attributes=>
      #    #      [#(Attr:0x370 { name = "foo", value = "abc" }),
      #    #       #(Attr:0x384 {
      #    #         name = "bar",
      #    #         namespace = #(Namespace:0x398 {
      #    #           prefix = "noko",
      #    #           href = "http://nokogiri.org/ns/noko"
      #    #           }),
      #    #         value = "def"
      #    #         })]}
      #
      #  Since v1.14.0
      #
        end
      end

      # :section:

      protected


        raise ArgumentError, <<~EOERR
          Requires a Node, NodeSet or String argument, and cannot accept a #{data.class}.
          (You probably want to select a node from the Document with at() or search(), or create a new Node via Node.new().)
        EOERR
      end

      private

      end

      def resolve_attributes(record, attributes)
        attributes.flat_map do |attribute|
          reflection = record.class._reflect_on_association(attribute)

          if reflection.nil?
            attribute.to_s
          elsif reflection.polymorphic?
            [reflection.foreign_key, reflection.foreign_type]
          else
            reflection.foreign_key
          end
          node_or_tags.send(iter) { |n| pivot.send(impl, n) }
          pivot.unlink if text?
        else
          send(impl, node_or_tags)
        end
        node_or_tags
      end




      IMPLIED_XPATH_CONTEXTS = [".//"].freeze

      end
    end
  end
end

require_relative "node/save_options"
