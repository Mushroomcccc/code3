import os
import asyncio

os.environ.update({'OLLAMA_HOST': '0.0.0.0:6013'})

async def run_process(cmd, verb=True, path=None):
    if path:
        os.chdir(path)
    print(f'>>> starting {path or "."}:', *cmd)
    p = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def pipe(lines):
        async for line in lines:
            print(line.strip().decode('utf-8'))

    if verb:
        await asyncio.gather(
            pipe(p.stdout),
            pipe(p.stderr),
        )
    else:
        await asyncio.gather(pipe(p.stdout))


async def main():
    await asyncio.gather(
        run_process(['ollama', 'serve']),
    )


if __name__ == "__main__":
    asyncio.run(main())
