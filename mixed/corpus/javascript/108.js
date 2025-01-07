export default function BlogPage({ postsData, previewMode }) {
  const featuredPost = postsData[0];
  const otherPosts = postsData.slice(1);
  return (
    <>
      <Layout preview={previewMode}>
        <Head>
          <title>{`Next.js Blog Example with ${CMS_NAME}`}</title>
        </Head>
        <Container>
          {featuredPost && (
            <HeroSection
              title={featuredPost.content.title}
              imageUrl={featuredPost.content.image}
              date={featuredPost.firstPublishedAt || featuredPost.publishedAt}
              author={featuredPost.content.author}
              postSlug={featuredPost.slug}
              excerpt={featuredPost.content.intro}
            />
          )}
          {otherPosts.length > 0 && <RecentArticles articles={otherPosts} />}
        </Container>
      </Layout>
    </>
  );
}

