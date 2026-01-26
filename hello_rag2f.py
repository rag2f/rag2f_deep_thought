import asyncio

from rag2f.core.rag2f import RAG2F


async def main():
    # Create an instance of RAG2F
    # The rag2f_deep_thought plugin is automatically loaded via entry point!
    rag2f = await RAG2F.create(plugins_folder="./plugins/")

    # Show loaded plugins
    print("\n📦 Loaded plugins:")
    for plugin_id in rag2f.morpheus.plugins:
        plugin = rag2f.morpheus.plugins[plugin_id]
        print(f"  - {plugin_id} (path: {plugin.path})")

    # Show registered hooks
    print("\n🪝 Registered hooks:")
    for hook_name, hooks in rag2f.morpheus.hooks.items():
        print(f"  - {hook_name}: {len(hooks)} handler(s)")

    # Show embedders registered via optimus_prime
    print("\n🧠 Registered embedders (OptimusPrime):")
    if hasattr(rag2f, "optimus_prime"):
        embedders = rag2f.optimus_prime.registry
        if embedders:
            for embedder_id, embedder in embedders.items():
                print(f"  - {embedder_id}: {embedder}")
        else:
            print("  No embedders registered.")
        # Show the default embedder
        try:
            default_embedder = rag2f.optimus_prime.get_default()
            print(f"\n⭐ Default embedder: {default_embedder}")
        except Exception as e:
            print(f"\n⚠️  No default embedder found: {e}")
    else:
        print("  Attribute 'optimus_prime' not found on rag2f.")

    # Try processing a sample text
    # Use execute_handle_text_foreground directly (Result Pattern)
    result = rag2f.johnny5.execute_handle_text_foreground("Hello World!")
    if result.is_ok():
        print(f"\n✅ Status: success\n📝 Track ID: {result.track_id}")
    else:
        print(
            f"\n❌ Status: error\n📝 Code: {result.detail.code}\n📝 Message: {result.detail.message}"
        )

    # check if duplicated
    result = rag2f.johnny5.execute_handle_text_foreground("Hello World!")
    if result.is_ok():
        print(f"\n✅ Status: success\n📝 Track ID: {result.track_id}")
    else:
        print(
            f"\n❌ Status: error\n📝 Code: {result.detail.code}\n📝 Message: {result.detail.message}"
        )


if __name__ == "__main__":
    asyncio.run(main())
